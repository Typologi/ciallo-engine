use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl Mesh {
    pub fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u16], label: &str) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-vertex-buffer")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}-index-buffer")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    pub fn bind<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn cube(device: &wgpu::Device, label: &str) -> Self {
        // 为了给每个面独立着色，这里按“每面 4 顶点”组织，共 24 顶点。
        let vertices = [
            // Front (+Z)
            Vertex {
                position: [-0.35, -0.35, 0.35],
                color: [1.0, 0.3, 0.3],
            },
            Vertex {
                position: [0.35, -0.35, 0.35],
                color: [1.0, 0.3, 0.3],
            },
            Vertex {
                position: [0.35, 0.35, 0.35],
                color: [1.0, 0.3, 0.3],
            },
            Vertex {
                position: [-0.35, 0.35, 0.35],
                color: [1.0, 0.3, 0.3],
            },
            // Back (-Z)
            Vertex {
                position: [0.35, -0.35, -0.35],
                color: [0.3, 1.0, 0.3],
            },
            Vertex {
                position: [-0.35, -0.35, -0.35],
                color: [0.3, 1.0, 0.3],
            },
            Vertex {
                position: [-0.35, 0.35, -0.35],
                color: [0.3, 1.0, 0.3],
            },
            Vertex {
                position: [0.35, 0.35, -0.35],
                color: [0.3, 1.0, 0.3],
            },
            // Left (-X)
            Vertex {
                position: [-0.35, -0.35, -0.35],
                color: [0.3, 0.5, 1.0],
            },
            Vertex {
                position: [-0.35, -0.35, 0.35],
                color: [0.3, 0.5, 1.0],
            },
            Vertex {
                position: [-0.35, 0.35, 0.35],
                color: [0.3, 0.5, 1.0],
            },
            Vertex {
                position: [-0.35, 0.35, -0.35],
                color: [0.3, 0.5, 1.0],
            },
            // Right (+X)
            Vertex {
                position: [0.35, -0.35, 0.35],
                color: [1.0, 1.0, 0.3],
            },
            Vertex {
                position: [0.35, -0.35, -0.35],
                color: [1.0, 1.0, 0.3],
            },
            Vertex {
                position: [0.35, 0.35, -0.35],
                color: [1.0, 1.0, 0.3],
            },
            Vertex {
                position: [0.35, 0.35, 0.35],
                color: [1.0, 1.0, 0.3],
            },
            // Top (+Y)
            Vertex {
                position: [-0.35, 0.35, 0.35],
                color: [0.3, 1.0, 1.0],
            },
            Vertex {
                position: [0.35, 0.35, 0.35],
                color: [0.3, 1.0, 1.0],
            },
            Vertex {
                position: [0.35, 0.35, -0.35],
                color: [0.3, 1.0, 1.0],
            },
            Vertex {
                position: [-0.35, 0.35, -0.35],
                color: [0.3, 1.0, 1.0],
            },
            // Bottom (-Y)
            Vertex {
                position: [-0.35, -0.35, -0.35],
                color: [1.0, 0.3, 1.0],
            },
            Vertex {
                position: [0.35, -0.35, -0.35],
                color: [1.0, 0.3, 1.0],
            },
            Vertex {
                position: [0.35, -0.35, 0.35],
                color: [1.0, 0.3, 1.0],
            },
            Vertex {
                position: [-0.35, -0.35, 0.35],
                color: [1.0, 0.3, 1.0],
            },
        ];

        let indices: [u16; 36] = [
            0, 1, 2, 0, 2, 3, // front
            4, 5, 6, 4, 6, 7, // back
            8, 9, 10, 8, 10, 11, // left
            12, 13, 14, 12, 14, 15, // right
            16, 17, 18, 16, 18, 19, // top
            20, 21, 22, 20, 22, 23, // bottom
        ];

        Self::new(device, &vertices, &indices, label)
    }
}
