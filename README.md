# krack-up

Cut a large solid into pieces that fit on a printer bed, and add pentagon dowels on the cuts. It writes STL meshes. It does not slice and it does not write gcode.

The defaults come from `JM-UPDATED_-_JIMMY_FULL_PRINT_3MF_PENTAGON_DOWELS.3mf`, a P1S project cut by hand:

- bed 256 × 256 × 250 mm (`--printer` picks another preset, `--bed 300x300x250` sets any size)
- 16 mm off the bed width and depth and 6 mm off the height for brim and clips (`--margin-xy`, `--margin-z`); a part turns a quarter turn when the bed is longer one way
- pentagon dowels, 10 mm long
- circumradius 10, 7.5, or 5 mm, largest that fits the joint
- 0.1 mm clearance in the hole
- one dowel every 100 mm across each cut face, and at least two dowels on every cut face
- joints between objects that were already cut apart in a 3MF get the same rule; faces that already have enough dowels are left alone. A joint is found by matching outlines, so two identical flat faces on different objects can be mistaken for one
- Bambu negative parts (cut connector sockets) are cut out of their object; modifier parts are ignored
- scale the model about its center before cutting (`--scale`, default 1)
- each piece turned so the flat joint is on the bed when that needs less support
- each part number engraved 0.6 mm deep on one of its joint faces, hidden once glued; `--no-labels` turns it off

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m krackup ui
.venv/bin/python -m krackup model.3mf -o model-krackup --every 100 --min-pins 2
.venv/bin/python -m krackup model.stl --bed 350x350x340 --margin-xy 10
.venv/bin/python -m krackup model.stl --info
```

Output:

- `parts/part_###.stl` — print these as exported, Z = 0 on the bed
- `dowels/pentagon_R*_L10.stl` — print the count in `ASSEMBLY.txt`
- `ASSEMBLY.txt` and `assembly.json` — which part numbers share a joint

Open the STLs in Bambu Studio and arrange them. krack-up stops there.

Tests:

```bash
.venv/bin/python -m unittest
```
