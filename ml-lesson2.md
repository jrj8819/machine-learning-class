# Pandas으로 데이터 가공하기

NumPy를 활용하여 복잡한 데이터 배열(n 차원)을 저장하고 가공할 수 있었다. Pandas는 NumPy를 기반으로 만들어진 새로운 패키지로서 DataFrame이라는 효율적인 자료구조를 제공한다.

- NumPy의 배열

```python
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
ar = iris.data
```

![](/Users/raejin/machine-learning-class/img/test 2019-06-18 20-23-09.png)

- DataFrame : 근본적으로 행과 열 레이블이 부착된 다차원 배열, 여러가지 타입의 데이터를 저장할 수 있으며 데이터 누락도 혀용, 데이터베이스와 스프레드시트 프로그램과 유사

```python

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()
```

![](/Users/raejin/machine-learning-class/img/12.png)



## Pandas 설치 및 사용

시스템에 Pandas를 설치하려면 먼저 NumPy가 설치되어 있어야한다. 일반적으로 NumPy를 np라는 별명으로 임포트 하는 것처럼 Pandas도 pd라는 별명으로 임포트하자

```python
import pandas as pd
```

 Pandas는 NumPy의 데이터를 다양한 형태의 객체(Series, DataFrame, Index)로 구조화하여 여러 유용한 도구와 메서드, 기능을 제공하는 역할을 한다.



## Series : 다양한 종류의 인덱스를 가진 NumPy 배열

Series 객체는 기본적으로 1차원 NumPy 배열과 비슷해 보이지만 차이는 인덱스에 있다. NumPy 배열에는 값에 접근하는데 사용되는 (암묵적인, 지정하지 않아도 자동으로 설정되는) 정수형 인덱스가 있고, **Pandas의 Series에는 객체 생성 시 명시적으로 인덱스를 지정**할 수 있다.

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
```

Series에서 values와 index 속성으로 데이터와 인덱스를 접근할 수 있다.

```python
data.values
data.index
```

별도로 인덱스를 지정하지 않으면 정수형 인덱스가 설정된다.

```python
data[1]
data[1:3]	
```

인덱스를 명시적으로 지정하자. 명시적으로 지정되는 인덱스는 숫자나 문자열이 될수도 있다. 데이터 부분([0.25, 0.5, 0.75, 1.0])다음에 index 속성으로 인덱스 배열([2, 5, 3, 7])을 지정하자. 

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
data[5]

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
data['b']
```



## 딕셔너리로 Series 만들기

딕셔너리를 활용하면 인덱스와 값을 분명하게 구분하여 Series를 생성할 수 있다. 다음과 같은 딕셔너리를 만들자

```python
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
```

딕셔너리의 키(California, Texas, …)는 인덱스가 되고 해당하는 값들은 데이터가 된다. 다음과 같이 Series 객체를 만들자

```python
population = pd.Series(population_dict)
population
```

딕셔너리의 키가 시리즈의 인덱스가 되어 접근하거나 슬라이스를 할 수 있다.

```python
population['California']
population['California':'Illinois']
```



## 예제 ) 다음의 데이터로 area라는 Series 객체를 만드세요.

![](/Users/raejin/machine-learning-class/img/python-programming-class:ML_lesson3.ipynb at master · jrj8819:python-programming-class 2019-06-18 21-12-29.png)



## DataFrame : 여러 Series가 모인 2차원의 배열

Series가 유연한 인덱스의 1차원 배열이라면 DataFrame은 유연한 행 인덱스와 열을 가진 2차원배열이라고 볼 수 있다. 즉 DataFrame은 여러 Series객체의 연속이다. 

앞에서 구성했던 population, area  Series객체를 합쳐 DataFrame 객체를 구성하자. 딕셔너리의 형태로 키는 열의 이름 값은 Series 객체가 들어간다.

```python
states = pd.DataFrame({'population': population,
                       'area': area})
states
```

![](/Users/raejin/machine-learning-class/img/03.01-Introducing-Pandas-Objects 2019-06-18 21-22-21.png)

Series  객체와 마찬가지로 DataFrame도 인덱스를 접근할 수 있는 index 속성이 있다. 또한 열의 이름을 모아놓은 column 속성을 가지고 있다.

```python
states.index
states.columns
```

