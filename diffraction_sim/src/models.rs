pub struct Spot {
    pub x: f64,
    pub y: f64,
    pub intensity: f64,
    pub sigma_major: f64, // Elongated axis
    pub sigma_minor: f64, // Transverse axis
    pub angle: f64,       // Rotation angle of the ellipse in radians
}

pub struct Fragment {
    pub spots: Vec<Spot>,
    pub volume_fraction: f64,
}
