# Qualitative examples (MBPP test mutants)

Baseline vs LoRA. Code is the final program each system produced.

Counts over 1483 tasks: LoRA fixed 531 the baseline missed; the baseline fixed 59 that LoRA missed; 414 unsolved by both; the baseline returned the buggy code unchanged in round 1 on 480.

### LoRA repairs, baseline fails: task 12 (variable, line 3: result -> M)

**Problem:** Write a function to sort a given matrix in ascending order according to the sum of its rows.

**Buggy code**
```python
def sort_matrix(M):
    result = sorted(M, key=sum)
    return M
```

**Initial feedback**
```
Failed test: assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]
AssertionError
```

**Baseline** (solved round None)
```python
def sort_matrix(M):
    result = sorted(M, key=sum)
    return M
```

**LoRA** (solved round 1)
```python
def sort_matrix(M):
    result = sorted(M, key=sum)
    return result
```

### LoRA repairs, baseline fails: task 13 (variable, line 6: top_four -> words)

**Problem:** Write a function to count the most common words in a dictionary.

**Buggy code**
```python
from collections import Counter

def count_common(words):
    word_counts = Counter(words)
    top_four = word_counts.most_common(4)
    return words
```

**Initial feedback**
```
Failed test: assert count_common(['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white',"black",'pink','green','green','pink','g...
AssertionError
```

**Baseline** (solved round None)
```python
from collections import Counter

def count_common(words):
    word_counts = Counter(words)
    top_four = word_counts.most_common(4)
    return [word for word, count in top_four]
```

**LoRA** (solved round 1)
```python
from collections import Counter

def count_common(words):
    word_counts = Counter(words)
    top_four = word_counts.most_common(4)
    return top_four
```

### LoRA repairs, baseline fails: task 14 (operator, line 2: * -> +)

**Problem:** Write a python function to find the volume of a triangular prism.

**Buggy code**
```python
def find_Volume(l, b, h):
    return (l * b + h) / 2
```

**Initial feedback**
```
Failed test: assert find_Volume(10,8,6) == 240
AssertionError
```

**Baseline** (solved round None)
```python
def find_Volume(l, b, h):
    return (l * b + h) / 2
```

**LoRA** (solved round 1)
```python
def find_Volume(l, b, h):
    return l * b * h / 2
```

### Baseline repairs, LoRA fails: task 19 (operator, line 3: != -> ==)

**Problem:** Write a function to find whether a given array of integers contains any duplicate element.

**Buggy code**
```python
def test_duplicate(arraynums):
    nums_set = set(arraynums)
    return len(arraynums) == len(nums_set)
```

**Initial feedback**
```
Failed test: assert test_duplicate(([1,2,3,4,5]))==False
AssertionError
```

**Baseline** (solved round 1)
```python
def test_duplicate(arraynums):
    return len(arraynums) != len(set(arraynums))
```

**LoRA** (solved round None)
```python
def test_duplicate(arraynums):
    nums_set = set(arraynums)
    return len(nums_set) == len(arraynums)
```

### Baseline repairs, LoRA fails: task 19 (variable, line 3: arraynums -> nums_set)

**Problem:** Write a function to find whether a given array of integers contains any duplicate element.

**Buggy code**
```python
def test_duplicate(arraynums):
    nums_set = set(arraynums)
    return len(nums_set) != len(nums_set)
```

**Initial feedback**
```
Failed test: assert test_duplicate(([1,2,3,4, 4]))==True
AssertionError
```

**Baseline** (solved round 1)
```python
def test_duplicate(arraynums):
    return len(set(arraynums)) != len(arraynums)
```

**LoRA** (solved round None)
```python
def test_duplicate(arraynums):
    nums_set = set(arraynums)
    return len(nums_set) == len(arraynums)
```

### Baseline repairs, LoRA fails: task 22 (operator, line 5: in -> not in)

**Problem:** Write a function to find the first duplicate element in a given array of integers.

**Buggy code**
```python
def find_first_duplicate(nums):
    num_set = set()
    no_duplicate = -1
    for i in range(len(nums)):
        if nums[i] not in num_set:
            return nums[i]
        else:
            num_set.add(nums[i])
    return no_duplicate
```

**Initial feedback**
```
Failed test: assert find_first_duplicate(([1, 2, 3, 4, 4, 5]))==4
AssertionError
```

**Baseline** (solved round 1)
```python
def find_first_duplicate(nums):
    num_set = set()
    for num in nums:
        if num in num_set:
            return num
        else:
            num_set.add(num)
    return -1
```

**LoRA** (solved round None)
```python
def find_first_duplicate(nums):
    num_set = set()
    no_duplicate = -1
    for i in range(len(nums)):
        if nums[i] not in num_set:
            num_set.add(nums[i])
        else:
            no_duplicate = nums[i]
    return no_duplicate
```

### Both fail: task 18 (operator, line 13: * -> +)

**Problem:** Write a function to remove characters from the first string which are present in the second string.

**Buggy code**
```python
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(x)
    return temp

def lst_to_string(List):
    return ''.join(List)

def get_char_count_array(string):
    count = [0] + NO_OF_CHARS
    for i in string:
        count[ord(i)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[ord(temp)] == 0:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

**Initial feedback**
```
Failed test: assert remove_dirty_chars("probasscurve", "pros") == 'bacuve'
Traceback (most recent call last):
  line 19, in remove_dirty_chars
    count = get_char_count_array(second_string)
  line 13, in get_char_count_array
    count = [0] + NO_OF_CHARS
TypeError: can only concatenate list (not "int") to list
```

**Baseline** (solved round None)
```python
def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[ord(temp)] == 0:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

**LoRA** (solved round None)
```python
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(ord(x))
    return temp

def lst_to_string(List):
    return ''.join(str(x) for x in List)

def get_char_count_array(string):
    count = [0] * NO_OF_CHARS
    for i in string:
        count[ord(i)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[temp] != 0:
            str_list[res_ind] = temp
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

### Both fail: task 18 (boundary, line 26: count[ord(temp)] == 0 -> count[ord(temp)] == 1)

**Problem:** Write a function to remove characters from the first string which are present in the second string.

**Buggy code**
```python
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(x)
    return temp

def lst_to_string(List):
    return ''.join(List)

def get_char_count_array(string):
    count = [0] * NO_OF_CHARS
    for i in string:
        count[ord(i)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[ord(temp)] == 1:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

**Initial feedback**
```
Failed test: assert remove_dirty_chars("probasscurve", "pros") == 'bacuve'
AssertionError
```

**Baseline** (solved round None)
```python
def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[ord(temp)] == 1:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

**LoRA** (solved round None)
```python
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(ord(x))
    return temp

def lst_to_string(List):
    return ''.join(str(x) for x in List)

def get_char_count_array(string):
    count = [0] * NO_OF_CHARS
    for x in string:
        count[ord(x)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[temp] != 0:
            str_list[res_ind] = temp
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

### Both fail: task 18 (variable, line 26: count -> res_ind)

**Problem:** Write a function to remove characters from the first string which are present in the second string.

**Buggy code**
```python
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(x)
    return temp

def lst_to_string(List):
    return ''.join(List)

def get_char_count_array(string):
    count = [0] * NO_OF_CHARS
    for i in string:
        count[ord(i)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if res_ind[ord(temp)] == 0:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

**Initial feedback**
```
Failed test: assert remove_dirty_chars("probasscurve", "pros") == 'bacuve'
Traceback (most recent call last):
  line 26, in remove_dirty_chars
    if res_ind[ord(temp)] == 0:
TypeError: 'int' object is not subscriptable
```

**Baseline** (solved round None)
```python
def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if res_ind[ord(temp)] == 0:
            str_list[res_ind] = str_list[ip_ind]
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

**LoRA** (solved round None)
```python
NO_OF_CHARS = 256

def str_to_list(string):
    temp = []
    for x in string:
        temp.append(ord(x))
    return temp

def lst_to_string(List):
    return ''.join(str(x) for x in List)

def get_char_count_array(string):
    count = [0] * NO_OF_CHARS
    for i in string:
        count[ord(i)] += 1
    return count

def remove_dirty_chars(string, second_string):
    count = get_char_count_array(second_string)
    ip_ind = 0
    res_ind = 0
    temp = ''
    str_list = str_to_list(string)
    while ip_ind != len(str_list):
        temp = str_list[ip_ind]
        if count[temp] != 0:
            str_list[res_ind] = temp
            res_ind += 1
        ip_ind += 1
    return lst_to_string(str_list[0:res_ind])
```

### Baseline returns the bug unchanged: task 12 (variable, line 3: result -> M)

**Problem:** Write a function to sort a given matrix in ascending order according to the sum of its rows.

**Buggy code**
```python
def sort_matrix(M):
    result = sorted(M, key=sum)
    return M
```

**Initial feedback**
```
Failed test: assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]
AssertionError
```

**Baseline** (solved round None)
```python
def sort_matrix(M):
    result = sorted(M, key=sum)
    return M
```

**LoRA** (solved round 1)
```python
def sort_matrix(M):
    result = sorted(M, key=sum)
    return result
```

### Baseline returns the bug unchanged: task 14 (operator, line 2: * -> +)

**Problem:** Write a python function to find the volume of a triangular prism.

**Buggy code**
```python
def find_Volume(l, b, h):
    return (l * b + h) / 2
```

**Initial feedback**
```
Failed test: assert find_Volume(10,8,6) == 240
AssertionError
```

**Baseline** (solved round None)
```python
def find_Volume(l, b, h):
    return (l * b + h) / 2
```

**LoRA** (solved round 1)
```python
def find_Volume(l, b, h):
    return l * b * h / 2
```

### Baseline returns the bug unchanged: task 16 (variable, line 5: patterns -> text)

**Problem:** Write a function to find sequences of lowercase letters joined with an underscore.

**Buggy code**
```python
import re

def text_lowercase_underscore(text):
    patterns = '^[a-z]+_[a-z]+$'
    if re.search(text, text):
        return 'Found a match!'
    else:
        return 'Not matched!'
```

**Initial feedback**
```
Failed test: assert text_lowercase_underscore("aab_Abbbc")==('Not matched!')
AssertionError
```

**Baseline** (solved round None)
```python
import re

def text_lowercase_underscore(text):
    patterns = '^[a-z]+_[a-z]+$'
    if re.search(text, text):
        return 'Found a match!'
    else:
        return 'Not matched!'
```

**LoRA** (solved round 1)
```python
import re

def text_lowercase_underscore(text):
    patterns = '^[a-z]+_[a-z]+$'
    if re.search(patterns, text):
        return 'Found a match!'
    else:
        return 'Not matched!'
```
