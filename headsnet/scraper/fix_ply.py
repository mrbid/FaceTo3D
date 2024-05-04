# https://github.com/mrbid
import bpy
import glob
import pathlib
from os import mkdir
from os.path import isdir
importDir = "ply/"
outputDir = "ply_norm/"
if not isdir(outputDir): mkdir(outputDir)

for file in glob.glob(importDir + "*.ply"):
    model_name = pathlib.Path(file).stem
    if pathlib.Path(outputDir+model_name+'.ply').is_file() == True: continue
    bpy.ops.wm.ply_import(filepath=file)
    bpy.ops.wm.ply_export(
                            filepath=outputDir+model_name+'.ply',
                            filter_glob='*.ply',
                            check_existing=False,
                            ascii_format=False,
                            export_selected_objects=False,
                            apply_modifiers=True,
                            export_triangulated_mesh=True,
                            export_normals=True,
                            export_uv=False,
                            export_colors='SRGB',
                            global_scale=1.0,
                            forward_axis='Y',
                            up_axis='Z'
                        )
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge()
    bpy.ops.outliner.orphans_purge()
    bpy.ops.outliner.orphans_purge()