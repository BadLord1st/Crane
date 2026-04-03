use crate::engine::runtime::{BackendRuntimeShim, RuntimeModel};

pub fn from_backend_shim(shim: BackendRuntimeShim) -> Box<dyn RuntimeModel> {
    Box::new(shim)
}
