import matplotlib.pyplot as plt
import numpy as np
import math


# average values of array
def average(array):
    average_list = []
    for i in range(0, len(array)):
        average_list.append(sum(array[:i+1]) / len(array[:i+1]))
    return average_list


# calc sigma from mu
def sigma(n):
    sigma_value = []

    for i in range(1, n+1):
        sigma_value.append(1/i)

    return sigma_value


# Help function (fo r calculate)
def p_ab(a, b_a, b_na):
    """
    :param a: P(A)
    :param b_a: P(B/A)
    :param b_na: P(B/!A)
    :return: P(A/B)
    """
    return (b_a*a)/(b_a*a+b_na*(1-a))


# shot function
def shot(a, b_a, b_na):
    return (b_a*a)/(b_a*a+b_na*(1-a))


# not shot function
def not_shot(a, b_a, b_na):
    return ((1-b_a)*a)/((1-b_a)*a+(1-b_na)*(1-a))


def normal_distribution(a, b):
    # Creating vectors X and Y
    x = np.linspace(0, 10, 500)
    y = (1/(a*(2*math.pi)**0.5))*math.e**(-((x-b)**2)/(2*a**2))
    return x, y


# -------------------CODE-------------------------- #

# Exercise 1 part a
def ex_1_a(a=0.5, rounds=10, result=[]):
    """
    :param a: P(A)
    :param rounds: number on returns
    :param result: list of results ordered by rounds
    :return: result
    """
    result.append(a)
    for i in range(0, rounds):
        result.append(p_ab(result[i], 0.6, 0.35))
    return result


# Exercise 1 part b
def ex_1_b(a=0.05, rounds=10, result=[]):
    """
        :param a: P(A)
        :param rounds: number on returns
        :param result: list of results ordered by rounds
        :return: result
        """
    result.append(a)
    for i in range(0, rounds):
        result.append(p_ab(result[i], 0.6, 0.35))
    return result


# Exercise 1 part b
def ex_1_c(a=0.5, rounds=10, shot_rounds=5):
    for i in range(0, rounds):
        if i > shot_rounds-1:
            a = shot(a, 0.6, 0.35)
        else:
            a = not_shot(a, 0.6, 0.35)
    return a


# Exercise 2 part a
def ex_2_a(row=[3.5, 6, 6, 7, 7, 4, 5]):
    mu = average(row)
    sigma_values = sigma(len(row))

    for i in range(0, len(row)):
        x, y = normal_distribution(sigma_values[i], mu[i])
        plt.plot(x, y)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('posterior distribution for '+str(i) + ' lifts')
        plt.show()




def main():
    print("EX 1 part A, answers:", ex_1_a())
    print("EX 1 part B, answers:", ex_1_b())
    print("EX 1 part C, answers:", ex_1_c())
    print("EX 2 part A, answers: Graph(", ex_2_a(), ")")


if __name__ == '__main__':
    main()
