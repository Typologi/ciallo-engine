struct MvpUniform {
    mvp: mat4x4f,
};

@group(0) @binding(0)
var<uniform> u_mvp: MvpUniform;

struct VsOut { // 顶点阶段输出给光栅化/片元阶段的数据结构
    @builtin(position) position: vec4f, // 裁剪空间位置（必需）
    @location(0) color: vec3f, // 自定义插值颜色，传给片元阶段的 location(0)
};

@vertex // 标记这是顶点着色器入口
fn vs_main(
    @location(0) position: vec3f, // 来自顶点缓冲的位置属性
    @location(1) color: vec3f, // 来自顶点缓冲的颜色属性
) -> VsOut {
    var out: VsOut; // 构造输出结构
    out.position = u_mvp.mvp * vec4f(position, 1.0); // 应用 MVP，把模型空间坐标变换到裁剪空间
    out.color = color; // 透传顶点颜色到片元阶段
    return out; // 返回给后续流水线
}

@fragment // 标记这是片元着色器入口
fn fs_main(in: VsOut) -> @location(0) vec4f { // 接收插值后的颜色输入
    return vec4f(in.color, 1.0); // 输出最终像素颜色（alpha=1）
}
