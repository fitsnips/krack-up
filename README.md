# krack-up

Cut a large solid into pieces that fit on a printer bed, and add pentagon dowels on the cuts. It writes STL meshes. It does not slice and it does not write gcode.

The defaults come from `JM-UPDATED_-_JIMMY_FULL_PRINT_3MF_PENTAGON_DOWELS.3mf`, a P1S project cut by hand:

- bed 256 × 256 × 250 mm
- pentagon dowels, 10 mm long
- circumradius 10, 7.5, or 5 mm, largest that fits the joint
- 0.1 mm clearance in the hole
- one dowel every 100 mm across each cut, and at least two dowels on every cut plane
- each piece turned so the flat joint is on the bed when that needs less support

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m krackup ui
.venv/bin/python -m krackup model.3mf -o model-krackup --every 100 --min-pins 2
.venv/bin/python -m krackup model.stl --info
```

Output:

- `parts/part_###.stl` — print these as exported, Z = 0 on the bed
- `dowels/pentagon_R*_L10.stl` — print the count in `ASSEMBLY.txt`
- `ASSEMBLY.txt` and `assembly.json` — which part numbers share a joint

Open the STLs in Bambu Studio and arrange them. krack-up stops there.
