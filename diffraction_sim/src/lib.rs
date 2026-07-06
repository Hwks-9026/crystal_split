mod models;
mod generator;
mod compositor;

use nalgebra::{Matrix3, Rotation3};
use rand::Rng;
use generator::generate_fragment;
use compositor::Detector;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn generate_sample(py: Python<'_>, img_size: u32) -> PyResult<(PyObject, Vec<PyObject>)> {
    let mut detector = compositor::Detector::new(img_size);
    let mut mask_images = Vec::new();

    let mut rng = rand::thread_rng();

    let camera_length = rng.gen_range(180.0..250.0);
    let pixel_size = 0.1;

    let a = rng.gen_range(40.0..120.0);
    let b = rng.gen_range(40.0..120.0);
    let c = rng.gen_range(40.0..120.0);

    let alpha = rng.gen_range(80.0_f64..120.0).to_radians();
    let beta  = rng.gen_range(80.0_f64..120.0).to_radians();
    let gamma = rng.gen_range(80.0_f64..120.0).to_radians();

    let cos_a = alpha.cos();
    let cos_b = beta.cos();
    let cos_g = gamma.cos();
    let sin_g = gamma.sin();

    let volume_factor = (
        1.0
        - cos_a * cos_a
        - cos_b * cos_b
        - cos_g * cos_g
        + 2.0 * cos_a * cos_b * cos_g
    ).sqrt();

    let v = a * b * c * volume_factor;

    // Reciprocal cell parameters
    let a_star = b * c * sin_g / v;
    let b_star = a * c * beta.sin() / v;
    let c_star = a * b * alpha.sin() / v;

    // Busing–Levy B matrix
    let b_matrix = Matrix3::new(
        a_star,
        b_star * cos_g,
        c_star * cos_b,

        0.0,
        b_star * sin_g,
        c_star * (cos_a - cos_b * cos_g) / sin_g,

        0.0,
        0.0,
        1.0 / c,
    );

    let b_factor = rng.gen_range(5.0..50.0);

    let num_fragments = rng.gen_range(2..=3);
    let mut fragments = Vec::new();


    for i in 0..num_fragments {
        // randomize 3D orientation
        let rotation = Rotation3::from_euler_angles(
            rng.gen_range(0.0..std::f64::consts::TAU),
            rng.gen_range(0.0..std::f64::consts::TAU),
            rng.gen_range(0.0..std::f64::consts::TAU),
        );
        
        // randomize physical volume
        let volume_fraction = if i == 0 { 1.0 } else { rng.gen_range(0.1..0.6) };

        let fragment = generate_fragment(
            b_matrix, rotation, volume_fraction, b_factor, camera_length, pixel_size, 1024 
        );
        
        fragments.push(fragment);
    }

    for fragment in fragments.iter() {
        detector.composite_fragment(fragment);
        
        let mask = detector.generate_binary_mask(fragment, img_size);
        mask_images.push(PyBytes::new(py, mask.as_raw()).to_object(py));
    }

    //detector.apply_physics_and_noise();
    let final_img = detector.to_composite_image(img_size);
    let py_img = PyBytes::new(py, final_img.as_raw()).to_object(py);

    Ok((py_img, mask_images))
}

#[pymodule]
fn diffraction_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_sample, m)?)?;
    Ok(())
}

