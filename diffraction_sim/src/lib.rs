mod models;
mod generator;
mod compositor;

use nalgebra::{Matrix3, Rotation3};
use rand::Rng;
use generator::generate_fragment;
use compositor::Detector;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyfunction]
fn generate_sample(py: Python<'_>, img_size: u32) -> PyResult<(PyObject, Vec<PyObject>, PyObject)> {
    let mut detector = compositor::Detector::new(img_size);
    let mut mask_images = Vec::new();
    let mut rng = rand::thread_rng();

    let camera_length = rng.gen_range(40.0..150.0);
    
    let pixel_size = 0.2; 
    let wavelength = 1.0; 

    let a = rng.gen_range(5.0..40.0);
    let b = rng.gen_range(5.0..40.0);
    let c = rng.gen_range(5.0..40.0);

    let system_roll = rng.gen_range(0..100);
    
    let (alpha, beta, gamma);
    if system_roll < 33 {
        alpha = 90.0_f64.to_radians();
        beta = rng.gen_range(90.0_f64..120.0_f64).to_radians();
        gamma = 90.0_f64.to_radians();
    } else if system_roll < 66 {
        alpha = rng.gen_range(70.0_f64..110.0_f64).to_radians();
        beta = rng.gen_range(70.0_f64..110.0_f64).to_radians();
        gamma = rng.gen_range(70.0_f64..110.0_f64).to_radians();
    } else {
        alpha = 90.0_f64.to_radians();
        beta = 90.0_f64.to_radians();
        gamma = 90.0_f64.to_radians();
    }

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

    let b_factor = rng.gen_range(1.0..10.0);
    
    let num_fragments = 2;
    let mut fragments = Vec::new();

    for i in 0..num_fragments {
        let rotation = Rotation3::from_euler_angles(
            rng.gen_range(0.0..std::f64::consts::TAU),
            rng.gen_range(0.0..std::f64::consts::TAU),
            rng.gen_range(0.0..std::f64::consts::TAU),
        );
        
        let volume_fraction = if i == 0 { 1.0 } else { rng.gen_range(0.1..0.6) };

        let fragment = generate_fragment(
            b_matrix, rotation, volume_fraction, b_factor, camera_length, pixel_size, wavelength, 1024 
        );
        
        fragments.push(fragment);
    }

    for fragment in fragments.iter() {
        detector.composite_fragment(fragment);
        let mask = detector.generate_binary_mask(fragment, img_size);
        mask_images.push(PyBytes::new_bound(py, mask.as_raw()).to_object(py));
    }
    detector.apply_physics_and_noise();
    let final_img = detector.to_composite_image(img_size);
    let py_img = PyBytes::new_bound(py, final_img.as_raw()).to_object(py);

    let metadata = PyDict::new_bound(py);
    metadata.set_item("a", a)?;
    metadata.set_item("b", b)?;
    metadata.set_item("c", c)?;
    metadata.set_item("alpha", alpha.to_degrees())?;
    metadata.set_item("beta", beta.to_degrees())?;
    metadata.set_item("gamma", gamma.to_degrees())?;
    metadata.set_item("camera_length", camera_length)?;
    metadata.set_item("pixel_size", pixel_size)?;
    metadata.set_item("wavelength", wavelength)?;

    Ok((py_img, mask_images, metadata.to_object(py)))
}

#[pymodule]
fn diffraction_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_sample, m)?)?;
    Ok(())
}
