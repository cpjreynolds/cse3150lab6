# CSE3150 - Lab 5

## Usage

```
lab5.out f(-1) f(0) f(1)    read initial adjacency matrices
lab5.out -c file            read .csg file
lab5.out -i                 read .csg file from stdin
```

### Note

`barbell.csg` contains the barbell graph from the assignment pdf in `.csg` format.
The left 'bell' contains nodes `1..7` and the right contains `9..16`.
The two 'bells' are connected by node `8` between nodes `1` and `9`.

Running `lab5.out -c barbell.csg` outputs the matrix demonstrating that there
are zero-cost paths starting from nodes `1..7` to all nodes `9..16`.

## Compilation

`make all` to compile both testing and `main()` binaries.

`make check` to compile and run tests.

`make run` to compile and execute `./lab5.out tmat-1.txt tmat0.txt tmat1.txt`

`make barbell` to compile and execute `./lab5.out -c barbell.csg`

### Note

To compile **without** support for terminal colors, append `NOCOLOR=1` to the
`make` command. 

**e.g.** `make run NOCOLOR=1`

# Comma-separated graph (csg) format

To ease the input of graphs into the program - and to have a little fun - I created an extremely simple graph description language to describe labelled digraphs with edge weights of {1, -1}.

## Description

A `.csg` file consists of one or more lines, each consisting of a comma-separated list of numeric vertex labels alternating with edge weights.

A `.csg` file describes a single graph.

Vertex labels cannot be negative, and graph order is defined as `max(V) - min(V)` for a vertex set `V`.

Ordering and line breaks do not affect the parsed graph, thus:

```
0,1,1
1,-1,2
```
and
```
0,1,1,-1,2
```

specify the same graph.

The symbols `+` and `-` may be used in lieu of comma-separators and explicit
edge weights. `{+, -}` maps to `{1, -1}`.

```
0,1,1,-1,2
```
and
```
0+1-2
```

specify the same graph.

## Specification

```ebnf
newline = ? U+000A ? .

nonzero_digit = "1" … "9" .
digit = "0" | nonzero_digit .
number = nonzero_digit { digit } .
nonnegative_integer = "0" | number .
integer = ["-"] nonnegative_integer .

vertex = nonnegative_integer .
edge_weight = "," integer "," .
alt_edge_weight = "+" | "-" .
edge = edge_weight | alt_edge_weight .

csg = line { newline line } [ newline ].

line = vertex edge vertex { edge vertex } .

```
