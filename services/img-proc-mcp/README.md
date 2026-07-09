# img-proc-mcp

An MCP server that exposes image-manipulation tools to the vision agent.

Tools operate on **S3 keys**, not image bytes: each reads an image from the
configured bucket, applies a transform, writes the result to a new key
colocated with the input, and returns that new key. Region-capable tools
(`blur`, `add_noise`) take an optional bounding box `[x1, y1, x2, y2]` and, when
given one, transform only that region and paste it back into the full image.

## Tools

| Tool | Args | Notes |
|------|------|-------|
| `rotate` | `image_key, angle` | whole image |
| `flip` | `image_key, mode` | `"horizontal"` / `"vertical"` |
| `blur` | `image_key, radius=2.0, box=None` | region-capable |
| `resize` | `image_key, width, height` | whole image |
| `crop` | `image_key, box` | returns just the region |
| `add_noise` | `image_key, amount=0.05, box=None` | salt-and-pepper, region-capable |

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
