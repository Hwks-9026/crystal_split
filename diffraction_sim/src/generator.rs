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
    img_size: u32,
) -> Fragment {
    let mut spots = Vec::new();
    let center = (img_size / 2) as f64;
    
    // X-ray Beam Physics Definitions
    let lambda: f64 = 0.9795;
    let k_in = Vector3::new(0.0, 0.0, 1.0 / lambda);

    // 1. Calculate the maximum physical radius (to the detector corners)
    let max_radius_px = center.hypot(center); 
    let max_radius_mm = max_radius_px * pixel_size;

    // 2. Calculate max scattering angle theta using trigonometry
    let two_theta_max = max_radius_mm.atan2(camera_length);
    let theta_max = two_theta_max / 2.0;

    // 3. Convert angle to maximum reciprocal space distance (q_max)
    let q_max = (2.0 * theta_max.sin()) / lambda;

    // 4. Extract the unit cell scale factor from the diagonal of the B-matrix
    let max_cell_edge = 1.0 / b_matrix[(0, 0)]; 
    
    // 5. Dynamically calculate the safe HKL bounding loop limit
    let hkl_limit = (max_cell_edge * q_max).ceil() as i32;
    
    // Dynamically adjust Ewald sphere proximity tolerance based on resolution packing
    let s_max = 0.0002 * (hkl_limit as f64 / 15.0).max(1.0); 
    let oscillation_range = 1.0f64.to_radians();

    let mut rng = rand::thread_rng();
    let exp_dist = Exp::new(1.0).unwrap();
    
    // Randomize a base fuzziness (mosaicity/beam divergence) for this specific fragment
    let crystal_mosaicity = rng.gen_range(1.1..2.0); 

    for h in -hkl_limit..=hkl_limit {
        for k in -hkl_limit..=hkl_limit {
            for l in -hkl_limit..=hkl_limit {
                if h == 0 && k == 0 && l == 0 { continue; }

                let hkl = Vector3::new(h as f64, k as f64, l as f64);
                let g = b_matrix * hkl;
                let v = rotation * g;

                if let Some(hit_theta) = check_intersection(v, k_in, s_max, oscillation_range) {
                    let hit_rot = Rotation3::from_axis_angle(&Vector3::y_axis(), hit_theta);
                    let k_out = k_in + (hit_rot * v);

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

                        // 1. Calculate radial distance from detector center
                        let dx = x_px - center;
                        let dy = y_px - center;
                        let r = dx.hypot(dy);
                        
                        // 2. The angle of the ellipse points radially outward
                        let radial_angle = f64::atan2(dy, dx);

                        // 3. Base mosaicity profile
                        let base_fuzz = crystal_mosaicity + (intensity.log10() * 0.25).max(0.0);

                        // 4. Apply Radial Dispersion: Stretch the major axis further out it goes
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

/// Helper function to calculate spot intensity using Wilson statistics and B-factor falloff
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

fn check_intersection(v: Vector3<f64>, k_in: Vector3<f64>, _s_max: f64, range: f64) -> Option<f64> {
    let lambda = 1.0 / k_in.norm();
    let v_norm_sq = v.norm_squared();
    let a = -v.x;
    let b = v.z;
    let c = -0.5 * lambda * v_norm_sq;
    let r = a.hypot(b);

    if r < c.abs() { return None; }

    let phi = f64::atan2(b, a);
    let asin_val = (c / r).asin();
    let theta1 = asin_val - phi;
    let theta2 = std::f64::consts::PI - asin_val - phi;

    let normalize = |mut angle: f64| {
        while angle < -std::f64::consts::PI { angle += 2.0 * std::f64::consts::PI; }
        while angle >  std::f64::consts::PI { angle -= 2.0 * std::f64::consts::PI; }
        angle
    };

    for &raw_theta in &[theta1, theta2] {
        let theta = normalize(raw_theta);
        if theta >= 0.0 && theta <= range { return Some(theta); }
    }
    None
}
