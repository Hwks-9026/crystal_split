use crate::models::{Spot, Fragment};
use nalgebra::{Matrix3, Rotation3, Vector3};
use rand::Rng;
use rand_distr::{Exp, Distribution};

pub fn generate_fragment(
    b_matrix: Matrix3<f64>,
    rotation: Rotation3<f64>,
    volume_fraction: f64,
    b_factor: f64,
    camera_length: f64,
    pixel_size: f64,
    wavelength: f64,
    img_size: u32,
) -> Fragment {
    let mut spots = Vec::new();
    let center = (img_size / 2) as f64;
    
    let k_in = Vector3::new(0.0, 0.0, 1.0 / wavelength);

    let max_radius_px = center.hypot(center); 
    let max_radius_mm = max_radius_px * pixel_size;

    let two_theta_max = max_radius_mm.atan2(camera_length);
    let theta_max = two_theta_max / 2.0;

    let q_max = (2.0 * theta_max.sin()) / wavelength;

    let max_cell_edge = 1.0 / b_matrix[(0, 0)]; 
    
    let hkl_limit = (max_cell_edge * q_max).ceil() as i32;
    
    let mut rng = rand::thread_rng();
    let exp_dist = Exp::new(1.0).unwrap();
    
    let crystal_mosaicity = rng.gen_range(1.1..2.0); 
    
    let s_max = rng.gen_range(0.002..0.008); 

    for h in -hkl_limit..=hkl_limit {
        for k in -hkl_limit..=hkl_limit {
            for l in -hkl_limit..=hkl_limit {
                if h == 0 && k == 0 && l == 0 { continue; }

                let hkl = Vector3::new(h as f64, k as f64, l as f64);
                
                let g = b_matrix * hkl;
                let v = rotation * g;

                // --- EXCITATION ERROR CHECK ---
                let distance_from_center = (v + k_in).norm();
                let ideal_radius = 1.0 / wavelength;
                let excitation_error = (distance_from_center - ideal_radius).abs();

                if excitation_error <= s_max {
                    let k_out = k_in + v;

                    let x_px = center + ((camera_length * (k_out.x / k_out.z)) / pixel_size);
                    let y_px = center + ((camera_length * (k_out.y / k_out.z)) / pixel_size);

                    if x_px >= 0.0 && x_px < img_size as f64 && y_px >= 0.0 && y_px < img_size as f64 {
                        let g_norm_sq = g.norm_squared();
                        
                        let intensity = simulate_intensity(
                            g_norm_sq, 
                            b_factor, 
                            volume_fraction, 
                            &mut rng, 
                            &exp_dist
                        );
                        
                        if intensity < 5.0 { continue; }

                        let dx = x_px - center;
                        let dy = y_px - center;
                        let r = dx.hypot(dy);
                        
                        let radial_angle = f64::atan2(dy, dx);

                        let base_fuzz = crystal_mosaicity + (intensity.log10() * 0.25).max(0.0);

                        let dispersion_factor = 1.0 + (r / img_size as f64) * 0.8; 
                        
                        spots.push(Spot {
                            x: x_px,
                            y: y_px,
                            intensity,
                            sigma_major: base_fuzz * dispersion_factor, 
                            sigma_minor: base_fuzz * 0.85,                
                            angle: radial_angle,
                        });
                    }
                }
            }
        }
    }

    Fragment { spots, volume_fraction }
}

fn simulate_intensity<R: Rng + ?Sized>(
    g_norm_sq: f64,
    b_factor: f64,
    volume_fraction: f64,
    rng: &mut R,
    exp_dist: &Exp<f64>,
) -> f64 {
    let random_modulator = exp_dist.sample(rng);
    let falloff = (-b_factor * g_norm_sq / 4.0).exp();
    let base_i = 15000.0 * volume_fraction;
    
    base_i * random_modulator * falloff
}
