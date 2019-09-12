
# Python Basics

## Python As A Calculator

Addition


```
5 + 5
```




    10



Substraction


```
5 - 5
```




    0



Multiplication


```
5 * 5
```




    25



Division


```
25 / 5
```




    5.0



Integer Division


```
25 // 5
```




    5




```
26 / 3
```




    8.666666666666666




```
26 // 3
```




    8



Modulo


```
26 % 3
```




    2




```
20 % 2
```




    0




```
21 % 2
```




    1



Exponentiation


```
4 ** 2
```




    16




```
3 ** 3
```




    27



How much is your $100 worth after 7 years?


```
100 * 1.1 ** 7
```




    194.87171000000012



## Variables & Types

### Variable
* Specific, case-sensitive name
* Call up value through variable name
* Suppose you want to save your height and weight, you're 1.79m tall and weigh 68.7 kg you can assign these two values to variables with an equal sign.
```python
height = 1.79
weight = 68.7
```

Every time you type the variable's name, you're asking Python to reference the actual value of the variable

### Types
* Interger
* Float
* String
* Boolean


```
# These are comments which start with #,
# comments are ignored by Python interpreter and are for humans

# Create a variable savings
savings = 100
```


```
print(savings)
```

    100



```
# Create a variable growth_multiplier

growth_multiplier = 1.1

# Calculate result

result = savings * growth_multiplier ** 7

print(result)
```

    194.87171000000012


### Other variable types
In the previous exercise, you worked with two Python data types:

`int`, or integer: a number without a fractional part. `savings`, with the value `100`, is an example of an integer.
`float`, or floating point: a number that has both an integer and fractional part, separated by a point. `growth_multiplier`, with the value `1.1`, is an example of a float.
Next to numerical data types, there are two other very common data types:

`str`, or string: a type to represent text. You can use single or double quotes to build a string.
`bool`, or boolean: a type to represent logical values. Can only be `True` or `False` (the capitalization is important!).

Create a new string, `desc`, with the value `"compound interest"`. \
Create a new boolean, `profitable`, with the value `True`.


```
# Create a variable desc
desc = "compound interest"

# Create a variable profitable
profitable = True
```

### Guess the type
To find out the type of a value or a variable that refers to that value, you can use the `type()`` function. Suppose you've defined a variable `a`, but you forgot the type of this variable. To determine the type of a, simply execute:

```python
type(a)
```

### Operations with other types
In Python, different functions behave differently on different data types.


When you sum two strings, for example, you'll get different behavior than when you sum two integers or two booleans.



* Calculate the product of `savings` and `growth_multiplier`. Store the result in `year1`.
* What do you think the resulting type will be? Find out by printing out the type of `year1`.
* Calculate the sum of `desc` and `desc` and store the result in a new variable `doubledesc`.
* Print out `doubledesc`


```
savings = 100
growth_multiplier = 1.1
desc = "compound interest"

# Assign product of growth_multiplier and savings to year1
year1 = savings * growth_multiplier

# Print the type of year1
print(type(year1))
print(year1)

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print(type(doubledesc))
print(doubledesc)
```

    <class 'float'>
    110.00000000000001
    <class 'str'>
    compound interestcompound interest


### Type conversion

Using the `+` operator to paste together two strings can be very useful in building custom messages.

Suppose, for example, that you've calculated the return of your investment and want to summarize the results in a string. Assuming the floats `savings` and `result` are defined, you can try something like this:
```
print("I started with $" + savings + " and now have $" + result + ". Awesome!")
```
This will not work, though, as you cannot simply sum strings and floats.

To fix the error, you'll need to explicitly convert the types of your variables. More specifically, you'll need `str()`, to convert a value into a string. `str(savings)``, for example, will convert the float `savings` to a string.

Similar functions such as `int()`, `float()` and `bool()` will help you convert Python values into any type.


```
# Definition of savings and result
savings = 100
result = 100 * 1.10 ** 7

# Fix the printout
print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)

```

    I started with $100 and now have $194.87171000000012. Awesome!
```
