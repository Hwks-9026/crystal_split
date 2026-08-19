import json
import logging
import io
from cctbx import sgtbx, uctbx
from dxtbx.model import ExperimentList
from dials.command_line.stills_process import phil_scope, Processor, do_import


def load_datasets(json_path="datasets.json"):
    with open(json_path) as f:
        return json.load(f)


def make_params(space_group=None, unit_cell=None, d_min=None, output_dir="."):
    """Build stills_process params with optional known symmetry.

    Args:
        space_group: e.g. "P4", "I4122", "P212121"
        unit_cell: e.g. "79,79,38,90,90,90"
        d_min: resolution cutoff for spotfinder filter (Angstroms)
        output_dir: output directory (default ".")

    Returns:
        params: extracted PHIL params ready for Processor
    """
    params = phil_scope.extract()
    params.output.composite_output = False
    params.output.output_dir = output_dir
    if space_group is not None:
        params.indexing.known_symmetry.space_group = sgtbx.space_group_info(space_group)
    if unit_cell is not None:
        vals = [float(x) for x in unit_cell.replace(" ", "").split(",")]
        params.indexing.known_symmetry.unit_cell = uctbx.unit_cell(vals)
    if d_min is not None:
        params.spotfinder.filter.d_min = d_min

    params.indexing.stills.refine_all_candidates=False
    params.indexing.stills.rmsd_min_px=5
    params.indexing.stills.ewald_proximal_volume_max=0.005
    params.indexing.stills.candidate_outlier_rejection=False
    params.indexing.stills.reflection_subsampling.enable=True
    params.spotfinder.filter.min_spot_size=3
    return params


def make_params_from_dataset(datasets, label):
    """Build stills_process params using known symmetry from a dataset entry.

    Args:
        datasets: dict loaded from datasets.json
        label: dataset key, e.g. "B"

    Returns:
        params: extracted PHIL params with space_group and unit_cell set
    """
    ds = datasets[label]
    sg = ds.get("indexing_space_group", ds["space_group"])
    d_min = ds.get("resolution", None)
    return make_params(space_group=sg, unit_cell=ds["unit_cell"], d_min=d_min)


def load_expt(image_path):
    """Load a single image file into an ExperimentList.

    Uses the same do_import function as dials.stills_process CLI,
    which correctly converts ImageSequence to ImageSet for stills.
    """
    return do_import(image_path, load_models=True)


def make_processor(params):
    """Create a single Processor instance to be reused for spotfinding and indexing."""
    return Processor(params, composite_tag="tmp")


def SP_spotfinder(expt, params, processor=None):
    """Run stills_process spot finding.

    Args:
        expt: ExperimentList
        params: stills_process PHIL params (from make_params)
        processor: optional Processor instance (created if not provided)

    Returns:
        (strong, processor): reflection table of strong spots and the Processor
    """
    if processor is None:
        processor = make_processor(params)
    strong = processor.find_spots(expt)
    return strong, processor


def SP_indexer(expt, strong, params, processor=None, verbose=True):
    """Run stills_process indexing.

    Args:
        expt: ExperimentList
        strong: reflection table from SP_spotfinder
        params: stills_process PHIL params (from make_params)
        processor: optional Processor instance (should be same one used for spotfinding)
        verbose: if True, capture and print DIALS log output on failure

    Returns:
        (experiments, indexed): indexed ExperimentList and reflection table
    """
    if processor is None:
        processor = make_processor(params)

    if not verbose:
        experiments, indexed = processor.index(expt, strong)
        return experiments, indexed

    # Capture DIALS logger output so we can show diagnostics on failure
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.DEBUG)
    dials_logger = logging.getLogger("dials")
    prev_level = dials_logger.level
    dials_logger.setLevel(logging.DEBUG)
    dials_logger.addHandler(handler)
    try:
        experiments, indexed = processor.index(expt, strong)
        return experiments, indexed
    except Exception as e:
        print("--- DIALS indexing log ---")
        print(log_buffer.getvalue())
        print("--- end log ---")
        raise
    finally:
        dials_logger.removeHandler(handler)
        dials_logger.setLevel(prev_level)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python sp_utils.py <image_path> <dataset_label> [datasets.json]")
        sys.exit(1)

    image_path = sys.argv[1]
    label = sys.argv[2]
    json_path = sys.argv[3] if len(sys.argv) > 3 else "datasets.json"

    datasets = load_datasets(json_path)
    params = make_params_from_dataset(datasets, label)
    expt = load_expt(image_path)
    processor = make_processor(params)

    print("Running spotfinder...")
    strong, processor = SP_spotfinder(expt, params, processor=processor)
    print(f"Found {len(strong)} strong spots")

    print("Running indexer...")
    experiments, indexed = SP_indexer(expt, strong, params, processor=processor)
    print(f"Indexed {len(indexed)} reflections")
