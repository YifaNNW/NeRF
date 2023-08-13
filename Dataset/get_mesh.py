import trimesh

def as_mesh(scene_or_mesh):
    """
    to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            list(
                scene_or_mesh.geometry.values()
            )
        )
    else:
        mesh = scene_or_mesh
    return mesh