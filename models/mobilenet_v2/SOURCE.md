# MobileNetV2 (1.0, 224) feature vector - tfjs graph model

- **Source:** https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1/model.json?tfjs-format=file
  (TF Hub `google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1`; redirects to Kaggle storage)
- **Downloaded:** 2026-10-31 by `scripts/fetch_model.mjs`
- **License:** Apache-2.0 (Google, MobileNetV2; https://www.apache.org/licenses/LICENSE-2.0)
- **Format:** TensorFlow.js graph model (`tf.loadGraphModel`), weights in 3 shards
- **Input:** `[-1, 224, 224, 3]` float32 RGB, values in **[0, 1]**
- **Output node:** `module_apply_default/hub_output/feature_vector/SpatialSqueeze` -> **1280-d** feature vector (shape `[N, 1280]`)

## SHA-256

| File | Bytes | SHA-256 |
|---|---|---|
| `model.json` | 84896 | `35c5046c717ec4ca9cec7893b68426356bfad17c59ed356a0083e0349b9bd139` |
| `group1-shard1of3.bin` | 4194304 | `4b4a2c2b6b39ffd895e73ce9bf83139bacd8fab81ddfd5878ac05f47109b11e7` |
| `group1-shard2of3.bin` | 4194304 | `8cb61cedfc032bdb0d3137487ded7bcd5d08b2c96e838804e1a42bde8b421e67` |
| `group1-shard3of3.bin` | 438664 | `a6a78bd9318835b28cfb8abcdc2d092380ac7cbfcbdf201334bbd330ff7cccd1` |

**Combined** (SHA-256 over `model.json` bytes followed by the three shards in order):

`6e59585358f6cde86ac8d6ed4359ad6c7748bb773f500f223e30795550d75c25`

The same value is stored in `model.sha256.json` for the app and the precompute script.
