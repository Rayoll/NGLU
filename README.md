## Noise-resilient graph-based multi-source urban land-use mapping: A case study of cities in three continents

#### Requirements:

- python >= 3.8
- pytorch >= 2.0.0
- dgl >= 1.1.2

#### Dataset:

The Multi-CUN dataset and datasets of the study areas are released at [Baidu Drive](https://pan.baidu.com/s/1IvcWj9r0SZ30mXG38c0mQg) [Code: u4nj]. You can download the dataset, unzip it, and place it in the ``data/datasets`` directory for experimental use. To use the constructed graph data, you need to download the ``cache.zip`` file and extract it to the root directory of the code.

The structure of the directory is as follows:

```
-cache
-ckpt
-data
	- datasets
	- graph_data
	- gt_data
	...
```

#### Evaluate NGLU model on the Multi-CUN dataset

```python
python main_Multi_CUN.py
```

#### Evaluate NGLU model on the five study areas

**1. Beijing center**

```
python main_Beijing.py
```

**2. Wuhan center**

```
python main_Wuhan.py
```

**3. Chengdu center**

```
python main_Chengdu.py
```

**4. Macao**

```
python main_Macao.py
```

**5. Helsinki**

```
python main_Helsinki.py
```

**6. Conducting land-use mapping on the Greater Sydney Region (GSR)**

```
python GSR_mapping.py
```

#### Ablation study

**1. [Exp1] baseline model**

```
python main_ab_exp.py --ab_status baseline
```

**2. [Exp 2] baseline model with the category-degree sensitive probability (CDP) strategy**

```
python main_ab_exp.py --ab_status w_CDP
```

**3. [Exp 3] baseline model with the graph modeling of similar parcels (GMS) and hybrid graph aggregation module (HGA)**

```
python main_ab_exp.py --ab_status w_GMS_HGA
```

**4. [Exp 4] baseline model with the parcel representation consistency constraint (PRC)**

```
python main_ab_exp.py --ab_status w_PRC
```

**5. [Exp 5] NGLU framework without the PRC**

```
python main_ab_exp.py --ab_status w_o_PRC
```

