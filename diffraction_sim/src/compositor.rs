use crate::models::Fragment;
use image::{imageops, ImageBuffer, Luma, Rgb};
use rand::Rng;
use rand_distr::{Normal, Poisson, Distribution};

pub struct Detector {
    pub size: u32,
    pub buffer: Vec<f64>,
}

impl Detector {
    pub fn new(size: u32) -> Self {
        Self { size, buffer: vec![0.0; (size * size) as usize] }
    }

    pub fn composite_fragment(&mut self, fragment: &Fragment) {
        for spot in &fragment.spots {
            Self::draw_gaussian(self.size, spot.x, spot.y, spot.intensity, spot.sigma_major, spot.sigma_minor, spot.angle, &mut self.buffer);
        }
    }

    pub fn generate_binary_mask(&self, fragment: &Fragment, target_size: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut mask_buffer = vec![0.0; (self.size * self.size) as usize];
        for spot in &fragment.spots {
            Self::draw_gaussian(self.size, spot.x, spot.y, 100.0, spot.sigma_major, spot.sigma_minor, spot.angle, &mut mask_buffer);
        }

        let mut img = ImageBuffer::from_pixel(self.size, self.size, Luma([0]));
        for y in 0..self.size {
            for x in 0..self.size {
                let idx = (y * self.size + x) as usize;
                if mask_buffer[idx] > 5.0 {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }
        
        imageops::resize(&img, target_size, target_size, imageops::FilterType::Nearest)
    }

    fn draw_gaussian(
        size: u32,
        cx: f64,
        cy: f64,
        intensity: f64,
        s_major: f64,
        s_minor: f64,
        angle: f64,
        target_buffer: &mut Vec<f64>
    ) {
        let radius = (s_major * 4.5).ceil() as i32; 
        let cx_idx = cx.round() as i32;
        let cy_idx = cy.round() as i32;

        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let s_maj_sq = s_major * s_major;
        let s_min_sq = s_minor * s_minor;

        let a = (cos_a * cos_a) / (2.0 * s_maj_sq) + (sin_a * sin_a) / (2.0 * s_min_sq);
        let b = (2.0 * angle).sin() * (1.0 / s_min_sq - 1.0 / s_maj_sq) / 4.0;
        let c = (sin_a * sin_a) / (2.0 * s_maj_sq) + (cos_a * cos_a) / (2.0 * s_min_sq);

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let px = cx_idx + dx;
                let py = cy_idx + dy;

                if px >= 0 && px < size as i32 && py >= 0 && py < size as i32 {
                    let x_diff = px as f64 - cx;
                    let y_diff = py as f64 - cy;

                    // Rotated 2D Gaussian evaluation
                    let exponent = a * x_diff * x_diff + 2.0 * b * x_diff * y_diff + c * y_diff * y_diff;
                    
                    if exponent < 10.0 { // Avoid calculating negligible exponents
                        let g_val = intensity * (-exponent).exp();
                        
                        // Optional: Mix in a small Lorentzian tail component (Pseudo-Voigt profile)
                        let lorentz_denom = 1.0 + (x_diff*x_diff + y_diff*y_diff) / (s_maj_sq);
                        let l_val = intensity * (1.0 / lorentz_denom);
                        
                        // 85% Gaussian core, 15% diffuse scattering tail
                        let profile_mix = 0.85 * g_val + 0.15 * l_val;

                        let idx = (py * size as i32 + px) as usize;
                        target_buffer[idx] += profile_mix;
                    }
                }
            }
        }
    }

    pub fn apply_physics_and_noise(&mut self) {
        let mut rng = rand::thread_rng();
        
        let center_x = (self.size as f64 / 2.0) + rng.gen_range(-2.0..2.0);
        let center_y = (self.size as f64 / 2.0) + rng.gen_range(-2.0..2.0);
        
        // --- RANDOMIZED EXPERIMENTAL ENVIRONMENT VARIABLES ---
        let read_noise_level = rng.gen_range(10.0..30.0);
        let read_noise = Normal::new(0.0, read_noise_level).unwrap();

        let ambient_fog = rng.gen_range(30.0..140.0);       
        let air_scatter_amp = rng.gen_range(500.0..1500.0); 
        
        let water_ring_amp = rng.gen_range(15.0..400.0);   
        let water_ring_center = rng.gen_range(330.0..370.0); 
        let water_ring_width = rng.gen_range(1200.0..3500.0);

        // Detector hardware properties
        let panel_size = 256;      // Size of a single sensor ASIC
        let gap_size = 8;          // Dead space between panels
        let max_well_capacity = 65535.0; // Detector saturation limit (16-bit)

        for y in 0..self.size {
            for x in 0..self.size {
                let idx = (y * self.size + x) as usize;

                let stride = panel_size + gap_size;
                if x % stride >= panel_size || y % stride >= panel_size {
                    self.buffer[idx] = -1.0; // Standard crystallographic flag for unmeasured pixels
                    continue;
                }

                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let r = dx.hypot(dy);

                if r < 35.0 || (dx > -4.0 && dx < 4.0 && dy > 0.0) {
                    self.buffer[idx] = 0.0; // Often tracked as 0 or -1
                    continue;
                }

                let air_scatter = air_scatter_amp / (r + 10.0); 
                let water_ring = water_ring_amp * (-(r - water_ring_center).powi(2) / water_ring_width).exp();
                
                self.buffer[idx] += ambient_fog + air_scatter + water_ring;
                let val = self.buffer[idx];

                if val > 0.0 {
                    let poisson = Poisson::new(val).unwrap_or(Poisson::new(1.0).unwrap());
                    self.buffer[idx] = poisson.sample(&mut rng) as f64;
                }

                let pixel_gain = rng.gen_range(0.98..1.02); // ±2% variation
                self.buffer[idx] *= pixel_gain;

                self.buffer[idx] += read_noise.sample(&mut rng);
                
                let defect_roll = rng.r#gen::<f64>();
                if defect_roll < 0.00005 {
                    self.buffer[idx] = 0.0; // Dead pixel
                } else if defect_roll < 0.0001 {
                    self.buffer[idx] = max_well_capacity; // Hot pixel
                }

                if self.buffer[idx] > max_well_capacity { 
                    self.buffer[idx] = max_well_capacity; 
                } else if self.buffer[idx] < 0.0 { 
                    self.buffer[idx] = 0.0; 
                }
            }
        }

        let num_zingers = rng.gen_range(0..8);
        for _ in 0..num_zingers {
            let zx = rng.gen_range(0..self.size - 1);
            let zy = rng.gen_range(0..self.size - 1);
            let intensity = rng.gen_range(20000.0..80000.0);
            
            let z_idx = (zy * self.size + zx) as usize;
            if self.buffer[z_idx] != -1.0 { self.buffer[z_idx] += intensity; }
            
            let z_idx_adjacent = (zy * self.size + (zx + 1)) as usize;
            if self.buffer[z_idx_adjacent] != -1.0 { self.buffer[z_idx_adjacent] += intensity * 0.3; }
        }
    }
    pub fn to_composite_image(&self, target_size: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut img = ImageBuffer::from_pixel(self.size, self.size, Luma([255]));
        let max_val = 1800.0; 

        for y in 0..self.size {
            for x in 0..self.size {
                let idx = (y * self.size + x) as usize;
                let intensity_scaled = ((self.buffer[idx] / max_val) * 255.0).clamp(0.0, 255.0);
                let inverted_val = (255.0 - intensity_scaled) as u8;
                
                img.put_pixel(x, y, Luma([inverted_val]));
            }
        }
        
        // Downsample using Lanczos3 for smooth, high-quality intensity gradients
        imageops::resize(&img, target_size, target_size, imageops::FilterType::Lanczos3)
    }
}
