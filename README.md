# krack-up

Cut a large solid into pieces that fit on a printer bed, and add pentagon dowels on the cuts. It writes STL meshes. It does not slice and it does not write gcode.

The defaults come from `JM-UPDATED_-_JIMMY_FULL_PRINT_3MF_PENTAGON_DOWELS.3mf`, a P1S project cut by hand:

- bed 256 × 256 × 250 mm (`--printer` picks another preset, `--bed 300x300x250` sets any size)
- 16 mm off the bed width and depth and 6 mm off the height for brim and clips (`--margin-xy`, `--margin-z`); a part turns a quarter turn when the bed is longer one way
- pentagon dowels, 10 mm long
- circumradius 10, 7.5, or 5 mm, largest that fits the joint
- 0.1 mm clearance in the hole
- one dowel every 100 mm across each cut face, and at least two dowels on every cut face
- joints between objects that were already cut apart in a 3MF get the same rule; faces that already have enough dowels are left alone. Halves of one Bambu cut (from `cut_information.xml`) are paired first; other joints are found by matching outlines, so two identical flat faces on different objects can be mistaken for one
- Bambu negative parts (cut connector sockets) are cut out of their object; modifier parts are ignored
- scale the model about its center before cutting (`--scale`, default 1)
- each piece turned so the flat joint is on the bed when that needs less support
- every joint face engraved 0.6 mm deep with its part number and the part it meets (`3-5` on part 3 where it meets part 5), hidden once glued; a small face gets only its own number. `--no-labels` turns it off

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
- `project.3mf` with `--3mf` — every part and dowel packed onto as few plates as fit, clear of the P1S/X1C wiper corner
  - for a Bambu printer (P1S, X1C, A1, A1 mini) it is a Bambu Studio project: printer, process, and filament are set and the plates are filled. The settings come from the input 3MF when it is a Bambu project for the same printer, otherwise from the Bambu Studio profiles installed on this computer (`KRACKUP_BAMBU_PROFILES` points at another `profiles` folder). Parts are spaced for the brim and supports in those settings
  - for other printers, or when no Bambu settings are found, it is a plain 3MF; open it with the same printer selected so the plates line up

krack-up stops at meshes; slice them yourself.

Tests:

```bash
.venv/bin/python -m unittest
```
