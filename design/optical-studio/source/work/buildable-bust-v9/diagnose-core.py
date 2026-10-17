try:exec(open('work/buildable-bust-v9/build-core-candidates.py').read())
except AssertionError:
 from OCP.BRepGProp import BRepGProp
 from OCP.GProp import GProp_GProps
 from OCP.Bnd import Bnd_Box
 from OCP.BRepBndLib import BRepBndLib
 e=TopExp_Explorer(a,TopAbs_SOLID)
 while e.More():
  g=GProp_GProps();BRepGProp.VolumeProperties_s(e.Current(),g);b=Bnd_Box();BRepBndLib.Add_s(e.Current(),b);print('SOLID',g.Mass(),b.Get());e.Next()
