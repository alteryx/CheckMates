# CheckMates

CheckMates is an Alteryx Open Source library which catches and warns of problems with your data and problem setup before modeling.

## Installation
```bash
python -m pip install checkmates
```
## Start
#### Load and validate example data
```python
from checkmates import (
    IDColumnsDataCheck
)
import pandas as pd

id_data_check_name = IDColumnsDataCheck.name
X_dict = {
        "col_1": [1, 1, 2, 3],
        "col_2": [2, 3, 4, 5],
        "col_3_id": [0, 1, 2, 3],
        "Id": [3, 1, 2, 0],
        "col_5": [0, 0, 1, 2],
        "col_6": [0.1, 0.2, 0.3, 0.4],
    }
X = pd.DataFrame.from_dict(X_dict)
id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
print(id_cols_check.validate(X))
```

#### Run AutoML
```python
from evalml.automl import AutoMLSearch
automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
automl.search()
```

## Next Steps

Read more about CheckMates on our [documentation page](#):

## Support

The CheckMates community is happy to provide support to users of CheckMates. Project support can be found in four places depending on the type of question:
1. For usage questions, use [Stack Overflow](#) with the `CheckMates` tag.
2. For bugs, issues, or feature requests start a [Github issue](#).
3. For discussion regarding development on the core library, use [Slack](#).
4. For everything else, the core developers can be reached by email at open_source_support@alteryx.com

## Built at Alteryx

**CheckMates** is an open source project built by [Alteryx](https://www.alteryx.com). To see the other open source projects we’re working on visit [Alteryx Open Source](https://www.alteryx.com/open-source). If building impactful data science pipelines is important to you or your business, please get in touch.

<p align="center">
  <a href="https://www.alteryx.com/open-source">
    <img src="https://alteryx-oss-web-images.s3.amazonaws.com/OpenSource_Logo-01.png" alt="Alteryx Open Source" width="800"/>
  </a>
</p>
