# From a CFD case to a raw UrbanWind case file

This document is for whoever runs new CFD cases and wants them in the
format of the UrbanWind dataset: either to score the trained IBO model on
them, or to train on them. It specifies

1. the raw case file the pipeline reads (`<estate>_<dir>.pt`);
2. the input files to export from the CFD (the VTK schema);
3. what `convert_cfd_case.py` does with them and what it checks;
4. how to validate a converted case and how to train on converted cases;
5. the conventions of the dataset that the converter reproduces on purpose.

Every statement about the dataset below was verified on its files.

## 1. The raw case file

One file per case and wind direction, `torch.save` of a plain dict, named
`<estate>_<dir>.pt` with `dir` in N, E, S, W (the wind direction the case
stands for). The frame is the **model frame**: metres, origin at the centre
of the 1,100 m × 1,100 m estate, ground at z = 0, the mean wind blowing
toward −y. Every direction of an estate is stored in this frame; the letter
only records which real-world direction it was.

| key | shape, dtype | content | unit, convention |
|---|---|---|---|
| `schema` | str | `hdb-3d-v3-physical` | |
| `case_name` | str | `<estate>_<dir>` | |
| `volume_pos` | [N, 3] float32 | **vertices** of the CFD volume mesh inside the learning region \|x\|, \|y\| ≤ 550 m, 0 ≤ z ≤ 160 m (bounds included) | m |
| `volume_fields` | [N, 5] float32 | U<sub>x</sub>, U<sub>y</sub>, U<sub>z</sub>, p, ν<sub>t</sub> at the vertices: the mean of the values of the cells sharing each vertex | m/s; p **kinematic** (p/ρ, m²/s²) and **including the hydrostatic term −g z**; ν<sub>t</sub> in m²/s, floored at 2·10⁻¹² |
| `volume_sdf` | [N] float32 | signed distance from each vertex to the building surface (the STL) | m, **negative outside** the buildings, 0 on the surface |
| `volume_sdf_grad` | [N, 3] float32 | gradient of `volume_sdf` | unit vectors; for a point outside a building it points toward the building |
| `surface_pos` | [S, 3] float32 | centres of the wall faces of the building patches (buildings only, no ground) | m |
| `surface_normals` | [S, 3] float32 | face normals | unit, pointing **into the fluid** (roofs: +z) |
| `surface_areas` | [S] float32 | face areas | m² |
| `surface_fields` | [S, 4] float32 | p, τ<sub>x</sub>, τ<sub>y</sub>, τ<sub>z</sub> on the faces | p as in the volume; wall shear stress kinematic (τ/ρ, m²/s²) |
| `stl_vertices`, `stl_faces` | [V, 3] float32, [F, 3] int64 | the building triangle mesh | m |
| `stl_centers`, `stl_areas`, `stl_face_normals` | [F, 3], [F], [F, 3] float32 | per triangle | m, m², unit outward |
| `stl_centroid`, `stl_bbox_min`, `stl_bbox_max`, `stl_bbox_span` | [3] float32 | of the STL vertices | m |
| `L_curator` | float | largest side of the STL bounding box | m |
| `z_max_building` | float | top of the tallest building (STL) | m |
| `U_REF`, `RHO`, `Q`, `G_GRAVITY` | float | 2.0, 1.225, ρ U<sub>REF</sub>², 9.81: the training inflow speed at 15 m, the air density, ρU², g | informational |
| `volume_field_names`, `volume_field_units`, `surface_field_names`, `surface_field_units` | list[str] | labels of the field columns | |
| `volume_bbox_min`, `volume_bbox_max` | [3] float32 | the learning region | m |
| `source` | dict | written by the converter: input files, rotation, pressure convention, checks | |

The pipeline reads seven keys: `volume_pos`, `volume_fields`,
`surface_pos`, `surface_fields` (only p), `surface_normals`, `volume_sdf`
and `volume_sdf_grad`. Everything else is carried for inspection and for
plotting (slices, the paper's figures). The sign of `volume_sdf` is flipped
on load, so inside the code the distance is positive outside.

Two conventions deserve emphasis:

- **The volume points are mesh vertices, with cell values averaged onto
  them.** The dataset was made by reading the OpenFOAM internal mesh with
  the VTK/ParaView reader, whose cell-to-point step takes the unweighted
  mean of the cells sharing a point. So the ground plane z = 0 carries
  points (and non-zero velocities), the hanging nodes of a refinement
  interface are present, and a vertex on a roof holds the mean of the fluid
  cells above it. The model was trained on these values; new cases are
  processed the same way.
- **Pressure.** The solver ran with gravity: the kinematic pressure has a
  slope of −9.81 m²/s² per metre of height. The training target is
  p + g z (the detrend in `norm_stats.json`), so a case without gravity or
  in Pa has to be brought to this convention first; the converter does
  (section 3).

Several directions of one estate must be **index-aligned rigid
rotations** of each other: the same mesh rotated, so that point i is the
same point in every file. The preprocessing detects the quarter turn
between directions and fails if no exact rotation maps one file onto the
other. One direction per estate is fine.

## 2. Input files (the VTK schema)

Export three things from the converged CFD, all in metres and in the same
frame (origin at the estate centre, ground at z = 0):

**A. The internal mesh** — a VTK unstructured grid (`.vtu` or legacy
`.vtk`) with the cells *and their connectivity*, and these **cell data**
arrays:

| array | components | unit |
|---|---|---|
| `U` | 3 | m/s |
| `p` | 1 | m²/s² (kinematic) or Pa, with or without gravity (section 3) |
| `nut` | 1 | m²/s |

Polyhedral cells must be written as polyhedra, not decomposed into
tetrahedra and pyramids: decomposition adds points that are not mesh
vertices. Point data in the file are ignored; the converter averages the
cell values itself, so that every case gets the dataset's convention. The
whole internal mesh may be exported; the converter crops it.

**B. The building wall patches** — VTK polygonal data (`.vtp` or legacy
`.vtk`), one or more files, holding the wall faces of the **buildings
only** (not the ground, not the domain boundaries), with **cell data**:

| array | components | unit |
|---|---|---|
| `p` | 1 | as in the volume |
| `wallShearStress` | 3 | kinematic (m²/s²) or Pa, consistent with `p` |

Face centres, areas and normals are computed from the polygons; the
winding order does not matter, the normals are oriented with the STL.

**C. The buildings as an STL** (or any triangle mesh trimesh reads): the
geometry the mesh was snapped to, watertight if at all possible. A
watertight mesh (every edge shared by exactly two triangles) lets the
converter repair the orientation of every building; an open one (open
bottoms, buildings sharing a wall) is accepted with its own orientation,
which is then checked against points that are certainly outside the
buildings.

**D. Metadata** on the converter's command line: the estate name, the wind
direction label, the axis the wind blows toward in the CFD frame, and the
pressure convention when it cannot be detected.

Array names other than `U`, `p`, `nut`, `wallShearStress` are passed with
`--u-name`, `--p-name`, `--nut-name`, `--wss-name`.

### Exporting from OpenFOAM

```bash
simpleFoam -postProcess -func wallShearStress -latestTime
foamToVTK -latestTime -fields '(U p nut wallShearStress)'
```

(replace `simpleFoam` by your solver). Write the internal mesh and the
building patches; keep the polyhedra undecomposed (ESI OpenFOAM: the
default, do not pass `-poly-decomp`; Foundation OpenFOAM: pass `-poly`).
Both the XML (`.vtu`/`.vtp`) and the legacy (`.vtk`) formats are read.
Patch files appear under `VTK/<case>_<time>/boundary/<patch>.*` (ESI) or
`VTK/<patch>/...` (Foundation).

In ParaView, open the case, select *internalMesh* and the building patches,
switch off *Decompose polyhedra* in the reader, and *File → Save Data* each
block (internal mesh as `.vtu`, patches as `.vtp`), writing all arrays.

## 3. Converting

```bash
python convert_cfd_case.py --volume case/internal.vtu --surface case/boundary/buildings.vtp --stl case/buildings.stl --estate my_estate --direction N --out raw
```

| option | meaning |
|---|---|
| `--volume`, `--surface` (several allowed), `--stl` | the three inputs |
| `--estate`, `--direction` | the output is `<out>/<estate>_<direction>.pt` |
| `--wind-toward` | axis the mean wind blows toward in the CFD frame: `-y` (default, the model frame), `+y`, `-x`, `+x`; the case is rotated by quarter turns to −y |
| `--p` | pressure convention of the input: `auto` (default), `kinematic-g`, `pa-g`, `kinematic`, `pa` |
| `--rho` | air density for Pa inputs, default 1.225 |
| `--wss-name none` | no wall shear stress in the input: zeros are written; such a file is for scoring, not for training |

What it does, in order:

1. **Point values.** The cell values of U, p, ν<sub>t</sub> are averaged
   onto the mesh vertices (unweighted mean of the cells sharing a vertex).
2. **Frame.** Positions, vectors, normals and the STL are rotated so that
   the wind blows toward −y.
3. **Crop.** Vertices with |x|, |y| ≤ 550 m and 0 ≤ z ≤ 160 m are kept,
   wall faces likewise; the STL stays whole.
4. **Pressure.** The slope of p against z over all vertices tells the
   convention: about −9.81 per metre is kinematic pressure with gravity,
   about −12.0 is Pa with gravity, about 0 is a run without gravity, where
   `--p kinematic` or `--p pa` must say the unit. The output is kinematic
   with the hydrostatic term: p/ρ for Pa inputs, p − g z for runs without
   gravity. Wall shear stress is divided by ρ for Pa inputs.
5. **Surface.** Face centres and areas with OpenFOAM's definitions (fan
   triangles about the vertex mean), normals oriented into the fluid using
   the STL: a face normal is flipped when it points against the outward
   normal of the STL at the closest point.
6. **Distance.** Exact signed distance from every vertex to the STL
   (closest point on the triangles through libigl's AABB tree; the sign
   from the angle-weighted pseudonormal of the closest face, edge or
   vertex, which is correct where a plain face normal is not) and its
   gradient; stored negative outside. Before that, the orientation of the
   STL is tested on vertices above or beyond the buildings.
7. **File.** The record of section 1, with `source` describing the inputs
   and the detected conventions.

Checks that stop the conversion (each message says what to do):

| check | reason |
|---|---|
| a cell array is missing or is point data only | the averaging convention needs the cell values |
| hexahedra mixed with tetrahedra/pyramids and no polyhedra | the polyhedra were decomposed on export |
| fewer than 100 vertices on z = 0, or fewer than 1,000 vertices in the region | the points are not mesh vertices (cell centres?), or the mesh is not in metres or not centred |
| \|U\| above 50 m/s | U is not in m/s |
| negative ν<sub>t</sub> | wrong array |
| the slope of p matches no convention, or contradicts `--p` | unit or gravity mismatch |
| after the rotation the mean wind of the upper layers does not blow toward −y | wrong `--wind-toward` |
| the STL spans more than 5 km | millimetres |
| more than 2% of the vertices certainly outside the buildings get a negative distance | parts of the STL are oriented inward (an open mesh; make it watertight so the orientation can be repaired per building) |
| the wall faces are a median > 0.5 m away from the STL | patches and STL describe different geometry, frames or units |

Warnings (printed, not fatal): tetrahedra present in a hex mesh, an STL
that is not watertight, horizontal faces at z = 0 in the patches (ground
included?), the tallest building above 160 m, an inflow speed at 13–17 m
far from the 2 m/s of the training data, zeros written for the wall shear
stress.

Runtime: about a minute for a mesh of 3–5 million vertices; the distance
queries take seconds.

## 4. Validating a converted case

```bash
python predict_new_case.py --raw-pt raw/my_estate_N.pt --weights path/to/weights --out outputs
```

predicts at the file's own points and scores the prediction against its
CFD fields. The footprints are extracted from the surface points exactly as
the dataset preprocessing does, so the features are those of a dataset
case. Outputs in `outputs/`:

- `<case>_metrics.json`: the relative L2 error of every channel in physical
  units and the z-score MSE — the paper's metrics (README → Results gives
  the test-set values to compare with);
- `<case>.npz` with `volume_fields` (prediction) and `volume_fields_cfd`
  (truth), same for the surface;
- `<case>_volume.vtk`, `<case>_surface.vtk`: point clouds with `U`, `p`,
  `nut` and `U_cfd`, `p_cfd`, `nut_cfd` for ParaView.

A converted case whose metrics are far above the paper's test-set values
usually has an input problem the checks cannot see: another inflow profile
or speed, a different mesh resolution near the buildings, or geometry that
the footprint extraction misreads (the printed number of buildings should
match the estate).

## 5. Training on converted cases

Put the files of all estates and directions in one folder (several
directions of one estate must be rotations of the same mesh, section 1),
then

```bash
python main.py preprocess --norm-stats path/to/norm_stats.json --raw-dir path/to/raw
```

as described in README → Running the experiments, followed by the manifest
and training commands. Reusing the released `norm_stats.json` keeps the
targets on the trained model's scale, which is what fine-tuning wants; a
new dataset trained from scratch may recompute its statistics. The
anomaly filter of the preprocessing compares field statistics across cases,
so a folder must not mix files with and without wall shear stress.

## 6. Dataset conventions the converter reproduces on purpose

These are properties of the training data. They are not what a CFD
engineer would choose, but the model learned them, so new cases follow
them.

- The volume points are mesh vertices with averaged cell values: the
  velocity on the ground plane is not zero, a vertex on a roof carries the
  pressure of the cells half a cell above it (about −ρ g Δz/2 relative to
  the roof), and on a plane where the horizontal cells coarsen upward the
  hanging nodes of the coarse cells carry a few tens of Pa of interpolated
  hydrostatic pressure more than the regular nodes.
- Every STL vertex coincides with a volume point of distance 0 (the mesh
  was snapped onto the geometry's features).
- ν<sub>t</sub> is floored at 2·10⁻¹² m²/s.
- The learning region is cut on the vertices, bounds included, so the
  planes x = ±550 m and the ground plane are present.

Two things of the old files are not reproduced: about 1% of the far-away
points of the dataset's distance field have the wrong sign (an edge case of
the method used then; the pipeline never reads those values, it recomputes
its own distance except inside building footprints), and their unit labels
say `Pa` for what is kinematic pressure. The converter writes correct
labels.
