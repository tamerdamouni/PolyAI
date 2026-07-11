# img-proc-mcp

An MCP server that exposes image-manipulation tools to the vision agent.

Tools operate on **S3 keys**, not image bytes: each reads an image from the
configured bucket, applies a transform, writes the result to a new key
colocated with the input, and returns that new key.

Every transform takes an optional bounding box `[x1, y1, x2, y2]`. When given
one, only that region is transformed and pasted back into the full image — so
the agent can target a specific detected object ("blur the second dog"). If the
transform changes the region's dimensions (e.g. a rotation), the result is
squeezed back to the box's exact size so it always fits the hole it came from.
The image's overall size and everything outside the box are untouched.

## Tools

| Tool | Args | Notes |
|------|------|-------|
| `rotate` | `image_key, angle, box=None` | region rotate is refit into the box |
| `flip` | `image_key, mode, box=None` | `"horizontal"` / `"vertical"` |
| `blur` | `image_key, radius=2.0, box=None` | Gaussian |
| `resize` | `image_key, width, height, box=None` | with a box: resample the region in place |
| `crop` | `image_key, box` | returns just the region |
| `add_noise` | `image_key, amount=0.05, box=None` | salt-and-pepper |

## Layout

- `transforms.py` — pure functions, PIL `Image` → PIL `Image`. No S3, no MCP.
- `app.py` — thin tool wrappers doing S3 read → transform → S3 write.

## Configuration

- `AWS_REGION` (default `us-east-1`)
- `AWS_S3_BUCKET` — bucket the tools read from and write to

## Run

Install dependencies:

```
pip install -r requirements.txt
```

Start the server (HTTP transport on port 9000):

```
python app.py
```

Inspect the registered tools interactively:

```
fastmcp dev app.py
```

## Test

```
python -m pytest -v
```

Runs the pure-transform unit tests and the tool tests (S3 is mocked in-memory,
so no AWS credentials or network are needed).
