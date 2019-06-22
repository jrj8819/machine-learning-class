# 피벗 테이블

피벗테이블(pivot table)은 표 형태의 데이터로 작업하는 스프레드시트와 다른 프로그램에서 일반적으로 볼 수 있는 작업이다. 입력값으로 간단한 열 단위의 데이터를 취하고 그 데이터에 대한 다차원 요약을 제공하는 테이블을 구성한다.

Seaborn 라이브러리에 내장된 데이터를 사용하자.

```
import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()
```



### 예제)

groupby를 활용하여 성별에 따른 평균 생존률을 구하세요. 조금 더 심화하여 성별과 좌석 등급별 생존율을 보고 싶다면?

![](./img/28.png)![](./img/29.png)



## 피벗 테이블 구문

```python
titanic.pivot_table('survived', index='sex', columns='class')
```

피벗테이블은 pivot_table() 메서드를 사용해 만들 수 있으며, 첫 인수에는 계산할 데이터의 열 이름을 준다. index 와 columns는 피벗테이블에서 구분할 그룹(sex, class)을 의미하는 열이름을 넣는다.



### 다단계 피벗 테이블

피벗테이블은 여러가지 옵션을 활용하여 데이터의 구간을 만들어 활용할 수 있다. 다음은 pd.cut() 함수를 사용하여 연령별로 구간을 만든 예시이다.

```python
age = pd.cut(titanic['age'], [0, 18, 80])
```

 pd.cut() 함수는 주어진 데이터를 자를 구간의 구분값(경계값)으로 나눈다. 위에서는 (0~18, 18~80)의 2구간으로 나누어 데이터를 구분해준다.

```python
titanic.pivot_table('survived', index=['sex', age], columns='class')
```

다음은 지불한 비용(fare)에 따라서도 분석해보자. 여기서는 pd.qcut() 함수를 활용한다.

```python
fare = pd.qcut(titanic['fare'], 2)
```

pd.qcut() 함수는 주어진 데이터를 N 개의 파트로 구분한다. 위의 코드에서는 운임을 2개의 파트로 구분하여 활용한다,

```python
titanic.pivot_table('survived', index=['sex', age], columns=[fare, 'class'])
```

### 

### 기타 피벗 테이블 옵션

```python
# Pandas 0.19 버전 기준 호출 시그니처
DataFrame.pivot_table(data, values=None, index=None, columns=None,
                      aggfunc='mean', fill_value=None, margins=False,
                      dropna=True, margins_name='All')
```

- fill_value : 누락된 값 채우기

- dropna : 누락된 값 지우기

- aggfunc : 집계에 사용할 연산을 지정(기본값으로는 평균이다, sum, mean, count, min, max 등을 쓸 수 있음)

  - 원하는 열을 다른 집계로 매핑한 딕셔너리를 지정할 수 도 있다.

  - ```python
    titanic.pivot_table(index='sex', columns='class',
                        aggfunc={'survived':sum, 'fare':'mean'})
    ```

- margins : 그룹별 총합을 계산

- margin_name : 홍합을 구한 열과 행의 이름을 지정한다.

  ```python
  titanic.pivot_table('survived', index='sex', columns='class', margins=True)
  ```



### 예제) 

다음과 같이 연대별 남녀 출생수를 구해보자. 데이터셋은 



![30](assets/30.png)