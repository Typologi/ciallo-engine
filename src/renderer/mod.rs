use std::sync::Arc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::camera::Camera;

mod mesh;

use self::mesh::{Mesh, Vertex};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MvpUniform {
    mvp: [[f32; 4]; 4],
}

/// 渲染器（第 9 天目标）：
/// - 初始化 wgpu Instance
/// - 基于窗口创建 Surface
///
/// 第 10 天扩展：
/// - 选择 Adapter
/// - 创建 Device / Queue
pub struct CialloRenderer {
    instance: wgpu::Instance, // wgpu 全局入口，负责和后端 API（Metal/Vulkan/DX）打交道
    surface: wgpu::Surface<'static>, // 与窗口绑定的显示目标（交换链入口）
    device: wgpu::Device, // 逻辑设备：创建 GPU 资源（buffer/texture/pipeline）
    queue: wgpu::Queue, // 提交命令到 GPU 执行
    surface_config: wgpu::SurfaceConfiguration, // 当前交换链配置
    surface_size: PhysicalSize<u32>, // 当前窗口像素尺寸
    render_pipeline: wgpu::RenderPipeline, // 绘制三角形用的渲染管线
    camera: Camera, // 第 18 天：接入 camera 模块作为 VP 来源
    orbit_target: Vec3, // 第 19 天：轨道相机注视点
    orbit_radius: f32, // 第 19 天：轨道半径
    orbit_yaw: f32, // 第 19 天：水平旋转角（弧度）
    orbit_pitch: f32, // 第 19 天：俯仰角（弧度）
    is_dragging: bool, // 第 19 天：是否正在鼠标拖拽
    last_cursor_pos: Option<(f64, f64)>, // 第 19 天：上一帧鼠标位置
    mvp_buffer: wgpu::Buffer, // 第 17 天：MVP uniform buffer
    mvp_bind_group: wgpu::BindGroup, // 第 17 天：将 MVP 绑定到 shader 的组
    mesh: Mesh, // 第 15 天：图元数据从 mesh 顶点/索引缓冲读取
    depth_texture: wgpu::Texture, // 第 20 天：深度纹理资源（跟窗口尺寸一致）
    depth_view: wgpu::TextureView, // 第 20 天：深度纹理视图（render pass 直接使用）
}

impl CialloRenderer {
    fn create_depth_texture(
        device: &wgpu::Device,
        size: PhysicalSize<u32>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth-texture"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        (depth_texture, depth_view)
    }

    pub async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size(); // 先记录窗口尺寸，后续用于配置 surface
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default()); // 创建 wgpu 入口对象
        let surface = instance
            .create_surface(window) // 把窗口包装成可显示的 surface
            .map_err(|err| format!("Failed to create wgpu surface: {err}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance, // 偏向高性能 GPU
                compatible_surface: Some(&surface), // 要求该 GPU 能渲染到当前 surface
                force_fallback_adapter: false, // 不强制使用软件/降级适配器
            })
            .await
            .map_err(|err| format!("Failed to request wgpu adapter: {err}"))?;
        let adapter_info = adapter.get_info(); // 读取 GPU 信息用于日志
        log::info!(
            "Using adapter: {} ({:?}, {:?})",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default()) // 创建逻辑设备与命令队列
            .await
            .map_err(|err| format!("Failed to request wgpu device: {err}"))?;
        let surface_caps = surface.get_capabilities(&adapter); // 查询 surface 在此 GPU 上支持的能力
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb) // 优先选 sRGB，颜色空间更符合显示器
            .or_else(|| surface_caps.formats.first().copied()) // 没有 sRGB 就退回第一个可用格式
            .ok_or("Surface reports no supported texture format".to_string())?;
        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Fifo)
        {
            wgpu::PresentMode::Fifo // 优先 FIFO（通常等同 VSync，稳定不撕裂）
        } else {
            *surface_caps
                .present_modes
                .first()
                .ok_or("Surface reports no supported present mode".to_string())? // 兜底：第一个可用模式
        };
        let alpha_mode = *surface_caps
            .alpha_modes
            .first()
            .ok_or("Surface reports no supported alpha mode".to_string())?; // 选择一个可用 alpha 合成模式
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // 该纹理将作为渲染目标使用
            format: surface_format, // 交换链纹理格式
            width: size.width.max(1), // 宽度至少为 1，避免最小化导致 0 尺寸非法
            height: size.height.max(1), // 高度至少为 1
            present_mode, // 帧呈现策略
            alpha_mode, // 窗口合成时的 alpha 策略
            view_formats: vec![], // 不声明额外视图格式
            desired_maximum_frame_latency: 2, // 期望帧排队深度，降低输入到显示延迟
        };
        surface.configure(&device, &surface_config); // 真正把 swapchain 配置到 surface 上
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle-shader"), // 调试标签
            source: wgpu::ShaderSource::Wgsl(include_str!("triangle.wgsl").into()), // 编译内嵌 WGSL
        });
        let mut camera = Camera::new(
            Vec3::new(1.8, 1.6, 2.6),
            Vec3::ZERO,
            Vec3::Y,
            45.0_f32.to_radians(),
            size.width.max(1) as f32 / size.height.max(1) as f32,
            0.1,
            10.0,
        );
        camera.set_aspect(size.width.max(1) as f32 / size.height.max(1) as f32);
        let mvp_uniform = Self::build_mvp_uniform(&camera);
        let mvp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mvp-uniform-buffer"),
            contents: bytemuck::bytes_of(&mvp_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mvp_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mvp-bind-group-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let mvp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mvp-bind-group"),
            layout: &mvp_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mvp_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle-pipeline-layout"), // 调试标签
            bind_group_layouts: &[&mvp_bind_group_layout], // 第 17 天：接入 MVP 的 bind group layout
            immediate_size: 0, // 不使用 immediate constants
        });
        let (depth_texture, depth_view) = Self::create_depth_texture(&device, size);
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("triangle-render-pipeline"), // 调试标签
            layout: Some(&pipeline_layout), // 绑定上面的 pipeline layout
            vertex: wgpu::VertexState {
                module: &shader, // 顶点阶段使用同一个 shader module
                entry_point: Some("vs_main"), // 顶点入口函数
                compilation_options: wgpu::PipelineCompilationOptions::default(), // 默认编译选项
                buffers: &[Vertex::layout()], // 绑定顶点布局：position + color
            },
            primitive: wgpu::PrimitiveState::default(), // 默认图元设置（三角形列表、背面剔除等默认值）
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(), // 默认无 MSAA
            fragment: Some(wgpu::FragmentState {
                module: &shader, // 片元阶段同样使用该 shader module
                entry_point: Some("fs_main"), // 片元入口函数
                compilation_options: wgpu::PipelineCompilationOptions::default(), // 默认编译选项
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format, // 输出到交换链格式
                    blend: Some(wgpu::BlendState::REPLACE), // 直接覆盖写入，不做混合
                    write_mask: wgpu::ColorWrites::ALL, // 写 RGBA 全通道
                })],
            }),
            multiview_mask: None, // 不启用 multiview
            cache: None, // 不使用 pipeline cache
        });
        let mesh = Mesh::cube(&device, "cube-mesh");

        Ok(Self {
            instance,
            surface,
            device,
            queue,
            surface_config,
            surface_size: size,
            render_pipeline,
            camera,
            orbit_target: Vec3::ZERO,
            orbit_radius: 3.5,
            orbit_yaw: 0.60,
            orbit_pitch: 0.45,
            is_dragging: false,
            last_cursor_pos: None,
            mvp_buffer,
            mvp_bind_group,
            mesh,
            depth_texture,
            depth_view,
        })
    }

    fn build_mvp_uniform(camera: &Camera) -> MvpUniform {
        let model = Mat4::IDENTITY;
        MvpUniform {
            mvp: (camera.view_projection_matrix() * model).to_cols_array_2d(),
        }
    }

    fn update_mvp_buffer(&mut self) {
        let mvp_uniform = Self::build_mvp_uniform(&self.camera);
        self.queue
            .write_buffer(&self.mvp_buffer, 0, bytemuck::bytes_of(&mvp_uniform));
    }

    fn sync_orbit_camera(&mut self) {
        let cos_pitch = self.orbit_pitch.cos();
        let position = self.orbit_target
            + Vec3::new(
                self.orbit_radius * cos_pitch * self.orbit_yaw.sin(),
                self.orbit_radius * self.orbit_pitch.sin(),
                self.orbit_radius * cos_pitch * self.orbit_yaw.cos(),
            );
        self.camera.set_look_at(position, self.orbit_target, Vec3::Y);
        self.update_mvp_buffer();
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return; // 最小化时跳过，0 尺寸不能 configure
        }
        if new_size == self.surface_size {
            return; // 尺寸没变化就不重复配置
        }

        self.surface_size = new_size; // 记录新尺寸
        self.surface_config.width = new_size.width; // 更新交换链宽度
        self.surface_config.height = new_size.height; // 更新交换链高度
        self.camera
            .set_aspect(new_size.width as f32 / new_size.height as f32);
        self.update_mvp_buffer();
        self.surface.configure(&self.device, &self.surface_config); // 重新配置 surface
        let (depth_texture, depth_view) = Self::create_depth_texture(&self.device, new_size);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    pub fn handle_mouse_input(&mut self, button: winit::event::MouseButton, state: winit::event::ElementState) {
        if button != winit::event::MouseButton::Left {
            return;
        }
        self.is_dragging = state == winit::event::ElementState::Pressed;
        if !self.is_dragging {
            self.last_cursor_pos = None;
        }
    }

    pub fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        if !self.is_dragging {
            self.last_cursor_pos = Some((x, y));
            return;
        }

        let (last_x, last_y) = match self.last_cursor_pos {
            Some(pos) => pos,
            None => {
                self.last_cursor_pos = Some((x, y));
                return;
            }
        };
        self.last_cursor_pos = Some((x, y));

        let dx = - (x - last_x) as f32;
        let dy = - (y - last_y) as f32;
        let rotate_speed = 0.008;

        self.orbit_yaw += dx * rotate_speed;
        self.orbit_pitch -= dy * rotate_speed;
        self.orbit_pitch = self.orbit_pitch.clamp(-1.45, 1.45);

        self.sync_orbit_camera();
    }

    pub fn render(&mut self) {
        if self.surface_size.width == 0 || self.surface_size.height == 0 {
            return; // 0 尺寸时不渲染，避免拿帧报错
        }

        match self.surface.get_current_texture() {
            Ok(frame) => {
                // 为当前帧纹理创建视图，render pass 通过它写颜色数据。
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                // 创建命令编码器，把这一帧的 GPU 命令录制进去。
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("clear-screen-encoder"),
                    });
                {
                    // 开启 render pass：先清屏，再画三角形。
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear-screen-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view, // 当前颜色附件是交换链帧视图
                            resolve_target: None, // 非 MSAA，不需要 resolve
                            depth_slice: None, // 非 3D 纹理切片
                            ops: wgpu::Operations {
                                // 在每帧开始时先清成深蓝灰背景。
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.08,
                                    g: 0.10,
                                    b: 0.14,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store, // 结束 pass 后保留颜色结果用于 present
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None, // 不做遮挡查询
                        timestamp_writes: None, // 不写 GPU 时间戳
                        multiview_mask: None, // 不启用 multiview
                    });
                    render_pass.set_pipeline(&self.render_pipeline); // 绑定三角形绘制管线
                    render_pass.set_bind_group(0, &self.mvp_bind_group, &[]); // 绑定 MVP uniform
                    self.mesh.bind(&mut render_pass); // 绑定 mesh 顶点/索引缓冲
                    render_pass.draw_indexed(0..self.mesh.index_count(), 0, 0..1); // 按索引绘制
                }
                self.queue.submit(std::iter::once(encoder.finish())); // 提交命令给 GPU 执行
                frame.present(); // 交换链 present，把这帧显示出来
            }
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config); // surface 丢失/过期时重配
            }
            Err(wgpu::SurfaceError::Timeout) => {
                log::warn!("Surface acquire timeout, skip this frame"); // 超时：跳过一帧
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("Surface out of memory"); // OOM：需要上层决定是否退出
            }
            Err(wgpu::SurfaceError::Other) => {
                log::error!("Surface acquire failed for unknown reason"); // 其他未知错误
            }
        }
        let _ = (&self.instance, &self.depth_texture); // 显式使用字段，避免“未读取字段”告警（当前阶段仅保留所有权）
    }
}