use glam::{Mat4, Vec3};

/// 相机（Camera）是“观察世界”的参数集合。
///
/// 你可以把它理解成一台虚拟摄像机：
/// - `position`：摄像机在世界中的位置（相机放在哪里）
/// - `target`：摄像机注视的点（朝哪里看）
/// - `up`：相机头顶方向（决定画面哪边是“上”）
/// - `fov_y_radians`：竖直方向视角（弧度，越大越“广角”）
/// - `aspect`：宽高比（width / height）
/// - `znear` / `zfar`：近裁剪面和远裁剪面（可见深度范围）
#[derive(Debug, Clone)]
pub struct Camera {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y_radians: f32,
    aspect: f32,
    znear: f32,
    zfar: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            // 60 度的竖直视角，适合作为初始值。
            fov_y_radians: 60.0_f32.to_radians(),
            // 16:9 是常见窗口比例。
            aspect: 16.0 / 9.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }
}

impl Camera {
    /// 创建一个可控参数的相机。
    ///
    /// 建议：
    /// - `fov_y_radians` 通常使用 `45° ~ 75°` 区间
    /// - `znear` 尽量不要太小（例如 0.0001），会影响深度精度
    /// - `zfar` 只设置到“够用”即可，避免深度分辨率浪费
    pub fn new(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            position,
            target,
            up,
            fov_y_radians,
            aspect,
            znear,
            zfar,
        }
    }

    /// 计算 `view_matrix`（视图矩阵）：把“世界空间”坐标变换到“相机空间”。
    ///
    /// 直观理解：
    /// - 世界里每个物体都要先被“搬到相机眼里”去描述；
    /// - 这一步之后，相机就相当于在原点 `(0,0,0)`，并朝固定方向看。
    ///
    /// 数学形式：
    /// - 常见写法是 `view = look_at(position, target, up)`；
    /// - 本质是在构造相机局部坐标轴（right/up/forward）并附带平移。
    ///
    /// 常见坑：
    /// - `up` 不能是零向量，也不要与“视线方向”完全平行；
    /// - 否则会导致基向量退化，矩阵不稳定。
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// 计算 `projection_matrix`（投影矩阵）：把“相机空间”压到“裁剪空间”。
    ///
    /// 这一步决定透视效果（近大远小）和可见深度范围：
    /// - `fov_y_radians`：控制画面张角（广角/长焦）
    /// - `aspect`：控制横向比例，错误会导致画面拉伸
    /// - `znear` / `zfar`：控制可见深度区间
    ///
    /// 注意：
    /// - `aspect` 必须大于 0
    /// - `znear` 必须大于 0，且小于 `zfar`
    /// - 这些值异常时，矩阵可能出现 NaN/Inf，渲染会异常
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh_gl(self.fov_y_radians, self.aspect, self.znear, self.zfar)
    }

    /// 计算 `view_projection_matrix`（VP 矩阵）。
    ///
    /// 公式：
    /// - `VP = projection * view`
    ///
    /// 用途：
    /// - 在渲染中可把两步预乘，减少每个顶点的重复计算准备；
    /// - 顶点着色器常见形式：`clip_pos = VP * world_pos`。
    ///
    /// 乘法顺序不能反：
    /// - `projection * view` 与 `view * projection` 含义不同，结果也不同。
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// 更新相机宽高比（`aspect = width / height`）。
    ///
    /// 典型调用时机是窗口 resize；当 `aspect <= 0` 时忽略本次更新。
    pub fn set_aspect(&mut self, aspect: f32) {
        if aspect > 0.0 {
            self.aspect = aspect;
        }
    }

    /// 直接更新相机观察参数，常用于交互控制（如鼠标轨道相机）。
    pub fn set_look_at(&mut self, position: Vec3, target: Vec3, up: Vec3) {
        self.position = position;
        self.target = target;
        self.up = up;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn view_matrix_moves_camera_position_to_origin() {
        let camera = Camera::default();
        let view = camera.view_matrix();

        // 视图变换后，相机自身位置应落到相机空间原点。
        let in_view = view.transform_point3(Vec3::new(0.0, 0.0, 5.0));
        assert!(in_view.abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn view_matrix_maps_target_to_negative_z_axis() {
        let camera = Camera::default();
        let view = camera.view_matrix();

        // 默认相机从 +Z 看向原点，目标点在相机空间中应位于 -Z 方向。
        let target_in_view = view.transform_point3(Vec3::ZERO);
        assert!(target_in_view.abs_diff_eq(Vec3::new(0.0, 0.0, -5.0), EPSILON));
    }

    #[test]
    fn projection_matrix_contains_only_finite_values() {
        let camera = Camera::default();
        let projection = camera.projection_matrix();

        // 投影矩阵若参数异常，常见表现是出现 NaN/Inf。
        for element in projection.to_cols_array() {
            assert!(element.is_finite());
        }
    }

    #[test]
    fn view_projection_matrix_matches_projection_mul_view() {
        let camera = Camera::default();
        let expected = camera.projection_matrix() * camera.view_matrix();
        let actual = camera.view_projection_matrix();

        assert!(actual.abs_diff_eq(expected, EPSILON));
    }
}