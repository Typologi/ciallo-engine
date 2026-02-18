pub use glam::{Mat4, Quat, Vec3};

/// 返回向量的长度（模）。
pub fn vec3_length(v: Vec3) -> f32 {
    v.length()
}

/// 返回单位化后的向量。
///
/// 当输入是零向量时，返回 `Vec3::ZERO`，避免出现 NaN。
pub fn vec3_normalize(v: Vec3) -> Vec3 {
    v.try_normalize().unwrap_or(Vec3::ZERO)
}

/// 返回两个向量的点积。
pub fn vec3_dot(a: Vec3, b: Vec3) -> f32 {
    a.dot(b)
}

/// 返回两个向量的叉积。
pub fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(b)
}

/// 返回两点之间的距离。
pub fn vec3_distance(a: Vec3, b: Vec3) -> f32 {
    a.distance(b)
}

/// 返回 4x4 单位矩阵。
///
/// 原理：
/// - 单位矩阵是矩阵乘法的中性元。
/// - `I * M = M` 且 `M * I = M`。
///
/// 作用：
/// - 作为组合变换时的起始值。
///
/// 常见位置：
/// - 构建 model/view/projection 乘法链。
/// - 作为 Transform 的默认初始值。
pub fn mat4_identity() -> Mat4 {
    Mat4::IDENTITY
}

/// 根据位移向量创建平移矩阵。
///
/// 原理：
/// - 使用齐次坐标（点的 w=1），把平移也表示成矩阵乘法。
///
/// 作用：
/// - 在不改变朝向的前提下移动物体或相机基点。
///
/// 常见位置：
/// - 构建模型矩阵（如 `T * R * S`）。
pub fn mat4_translation(t: Vec3) -> Mat4 {
    Mat4::from_translation(t)
}

/// 创建非均匀缩放矩阵。
///
/// 原理：
/// - x/y/z 三个轴可独立缩放。
/// - `(sx, sy, sz)` 分别控制三个轴的拉伸比例。
///
/// 作用：
/// - 进行统一缩放或按轴缩放。
///
/// 常见位置：
/// - 物体局部变换阶段（进入世界空间前）。
pub fn mat4_scale(s: Vec3) -> Mat4 {
    Mat4::from_scale(s)
}

/// 创建绕 X 轴旋转（弧度制）的矩阵。
///
/// 原理：
/// - 在右手坐标系下，按 X 轴正方向旋转。
///
/// 作用：
/// - 常用于 Pitch（俯仰）这类上下转动。
///
/// 常见位置：
/// - 相机或物体的局部旋转。
pub fn mat4_rotation_x(rad: f32) -> Mat4 {
    Mat4::from_rotation_x(rad)
}

/// 创建绕 Y 轴旋转（弧度制）的矩阵。
///
/// 原理：
/// - 在右手坐标系下，按 Y 轴正方向旋转。
///
/// 作用：
/// - 常用于 Yaw（偏航）这类左右转向。
///
/// 常见位置：
/// - 角色朝向与相机左右转动。
pub fn mat4_rotation_y(rad: f32) -> Mat4 {
    Mat4::from_rotation_y(rad)
}

/// 创建绕 Z 轴旋转（弧度制）的矩阵。
///
/// 原理：
/// - 在右手坐标系下，按 Z 轴正方向旋转。
///
/// 作用：
/// - 常用于 Roll（翻滚）或 2D 平面旋转。
///
/// 常见位置：
/// - 精灵/画布式旋转、调试可视化旋转。
pub fn mat4_rotation_z(rad: f32) -> Mat4 {
    Mat4::from_rotation_z(rad)
}

/// 矩阵乘法辅助函数。
///
/// 原理：
/// - 乘法顺序非常重要，矩阵组合不满足交换律。
/// - 在常见写法中，`a * b` 作用在向量时通常意味着先应用 `b` 再应用 `a`。
///
/// 作用：
/// - 组合 local/world/view/projection 等不同空间变换。
///
/// 常见位置：
/// - `model = T * R * S`，`mvp = P * V * M`。
pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    a * b
}

/// 返回矩阵的逆矩阵。
///
/// 原理：
/// - 逆矩阵用于“撤销”一次可逆变换。
/// - 对有效仿射变换，`M * inverse(M) ~= I`。
///
/// 作用：
/// - 将世界空间坐标还原回局部空间。
///
/// 常见位置：
/// - 相机视图矩阵、空间坐标来回转换。
pub fn mat4_inverse(m: Mat4) -> Mat4 {
    m.inverse()
}

/// 使用矩阵 `m` 变换一个位置（点）。
///
/// 原理：
/// - 点可看作齐次坐标 `(x, y, z, 1)`。
/// - 因为 w=1，平移会对点生效。
///
/// 作用：
/// - 在不同空间之间变换顶点/位置。
///
/// 常见位置：
/// - local -> world、world -> view 的位置转换。
pub fn mat4_transform_point(m: Mat4, p: Vec3) -> Vec3 {
    m.transform_point3(p)
}

/// 使用矩阵 `m` 变换一个方向向量。
///
/// 原理：
/// - 方向可看作齐次坐标 `(x, y, z, 0)`。
/// - 因为 w=0，平移不会影响方向向量。
///
/// 作用：
/// - 对轴向、法线、速度方向做旋转/缩放。
///
/// 常见位置：
/// - 基向量更新、移动方向计算。
pub fn mat4_transform_vector(m: Mat4, v: Vec3) -> Vec3 {
    m.transform_vector3(v)
}

/// 使用平移、旋转、缩放组合出 TRS 变换矩阵。
///
/// 原理：
/// - 该函数固定引擎中的组合约定，避免后续模块各写各的。
/// - 这里采用 `T * R * S`，语义是先局部缩放，再旋转，最后平移到目标位置。
///
/// 作用：
/// - 从 Transform 组件（T/R/S）直接构建模型矩阵。
///
/// 常见位置：
/// - 场景节点变换、后续 `Transform::local_matrix()`。
pub fn mat4_compose_trs(translation: Vec3, rotation: Mat4, scale: Vec3) -> Mat4 {
    mat4_translation(translation) * rotation * mat4_scale(scale)
}

/// 物体在父节点坐标系下的局部变换（TRS）。
///
/// 语义：
/// - `translation`：把局部原点放到父空间中的目标位置。
/// - `rotation`：在局部原点旋转当前节点朝向。
/// - `scale`：在局部原点按轴缩放当前节点尺寸。
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// 返回从 local 空间到 parent 空间的局部矩阵。
    ///
    /// 组合约定：
    /// - 写法是 `T * R * S`，对应先缩放、再旋转、最后平移。
    pub fn local_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// 返回从 local 空间到 world 空间的全局矩阵。
    ///
    /// 规则：
    /// - 没有父节点时，`global = local`。
    /// - 有父节点时，`global = parent_global * local`。
    pub fn global_matrix(&self, parent_global: Option<Mat4>) -> Mat4 {
        let local = self.local_matrix();
        match parent_global {
            Some(parent) => parent * local,
            None => local,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn vec3_length_and_normalize_work() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((vec3_length(v) - 5.0).abs() < EPSILON);

        let normalized = vec3_normalize(v);
        assert!(normalized.abs_diff_eq(Vec3::new(0.6, 0.8, 0.0), EPSILON));
        assert!((normalized.length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn vec3_dot_and_cross_follow_right_hand_rule() {
        let x = Vec3::X;
        let y = Vec3::Y;
        let z = Vec3::Z;

        assert!((vec3_dot(x, y) - 0.0).abs() < EPSILON);
        assert!((vec3_dot(x, x) - 1.0).abs() < EPSILON);
        assert!(vec3_cross(x, y).abs_diff_eq(z, EPSILON));
    }

    #[test]
    fn vec3_distance_is_correct() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 3.0);

        assert!((vec3_distance(a, b) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn mat4_identity_keeps_point_unchanged() {
        let p = Vec3::new(1.0, -2.0, 3.5);
        let out = mat4_transform_point(mat4_identity(), p);
        assert!(out.abs_diff_eq(p, EPSILON));
    }

    #[test]
    fn translation_affects_point_but_not_direction() {
        let t = mat4_translation(Vec3::new(10.0, 0.0, 0.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        let direction = Vec3::X;

        let moved_point = mat4_transform_point(t, point);
        let same_direction = mat4_transform_vector(t, direction);

        assert!(moved_point.abs_diff_eq(Vec3::new(11.0, 2.0, 3.0), EPSILON));
        assert!(same_direction.abs_diff_eq(direction, EPSILON));
    }

    #[test]
    fn rotation_z_ninety_turns_x_to_y() {
        let r = mat4_rotation_z(FRAC_PI_2);
        let x = Vec3::X;
        let y = mat4_transform_vector(r, x);
        assert!(y.abs_diff_eq(Vec3::Y, EPSILON));
    }

    #[test]
    fn matrix_multiplication_order_changes_result() {
        let p = Vec3::new(1.0, 0.0, 0.0);
        let t = mat4_translation(Vec3::new(5.0, 0.0, 0.0));
        let s = mat4_scale(Vec3::splat(2.0));

        let t_then_s = mat4_mul(t, s);
        let s_then_t = mat4_mul(s, t);

        let a = mat4_transform_point(t_then_s, p);
        let b = mat4_transform_point(s_then_t, p);

        assert!(!a.abs_diff_eq(b, EPSILON));
    }

    #[test]
    fn inverse_undoes_transform() {
        let m = mat4_compose_trs(
            Vec3::new(3.0, -2.0, 1.0),
            mat4_rotation_y(FRAC_PI_2),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let inv = mat4_inverse(m);
        let p = Vec3::new(0.5, 1.0, -3.0);

        let world = mat4_transform_point(m, p);
        let recovered = mat4_transform_point(inv, world);

        assert!(recovered.abs_diff_eq(p, 1e-4));
    }

    #[test]
    fn transform_default_is_identity() {
        let t = Transform::default();
        assert!(t.local_matrix().abs_diff_eq(Mat4::IDENTITY, EPSILON));
    }

    #[test]
    fn transform_local_matrix_translates_points() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            ..Default::default()
        };

        let moved = t.local_matrix().transform_point3(Vec3::ZERO);
        assert!(moved.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), EPSILON));
    }

    #[test]
    fn transform_local_matrix_rotates_direction() {
        let t = Transform {
            rotation: Quat::from_rotation_y(FRAC_PI_2),
            ..Default::default()
        };

        let dir = t.local_matrix().transform_vector3(Vec3::X);
        assert!(dir.abs_diff_eq(-Vec3::Z, EPSILON));
    }

    #[test]
    fn transform_local_matrix_scales_vector() {
        let t = Transform {
            scale: Vec3::new(2.0, 3.0, 4.0),
            ..Default::default()
        };

        let out = t.local_matrix().transform_vector3(Vec3::new(1.0, 1.0, 1.0));
        assert!(out.abs_diff_eq(Vec3::new(2.0, 3.0, 4.0), EPSILON));
    }

    #[test]
    fn transform_global_matrix_without_parent_matches_local() {
        let t = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_z(FRAC_PI_2),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };

        let local = t.local_matrix();
        let global = t.global_matrix(None);
        assert!(global.abs_diff_eq(local, EPSILON));
    }

    #[test]
    fn transform_global_matrix_with_parent_propagates_hierarchy() {
        let parent = Transform {
            translation: Vec3::new(5.0, 0.0, 0.0),
            ..Default::default()
        };
        let child = Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            ..Default::default()
        };

        let parent_global = parent.global_matrix(None);
        let child_global = child.global_matrix(Some(parent_global));

        let child_origin_in_world = child_global.transform_point3(Vec3::ZERO);
        assert!(child_origin_in_world.abs_diff_eq(Vec3::new(5.0, 2.0, 0.0), EPSILON));
    }

    #[test]
    fn transform_global_matrix_works_for_grandparent_chain() {
        let grandparent = Transform {
            translation: Vec3::new(1.0, 0.0, 0.0),
            ..Default::default()
        };
        let parent = Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            ..Default::default()
        };
        let child = Transform {
            translation: Vec3::new(0.0, 0.0, 3.0),
            ..Default::default()
        };

        let grandparent_global = grandparent.global_matrix(None);
        let parent_global = parent.global_matrix(Some(grandparent_global));
        let child_global = child.global_matrix(Some(parent_global));

        let child_origin_in_world = child_global.transform_point3(Vec3::ZERO);
        assert!(child_origin_in_world.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), EPSILON));
    }
}