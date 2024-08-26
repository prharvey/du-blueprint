# du-blueprint

A tool for generating Dual Universe blueprint files from models.

Quick start:
```
du-blueprint generate --auto --core=l my_model.obj my_blueprint.blueprint
```

The only supported format at the moment is `.obj`. For good results, use a manifold mesh.
For best results, take into account in game voxel limitations when making your model.

This tool is very much in the "make it work" stage of development. There are a lot of
easy improvements that can be made, so PRs are welcome. Just let me know if you are working
on something beforehand.

Right now the voxelization process is pretty naive and unoptimized, and just throws threads
at the problem.

## FAQ

### Q. My model has weird floating boxes.

You tried to import a non-manifold mesh. The voxelizer tries it's best to account
for this, but it isn't perfect.

### Q. How can I make my mesh manifold?

Blender. Search for a tutorial on the "3D-Print Toolbox" addon; this is a common problem
with 3D printing.
