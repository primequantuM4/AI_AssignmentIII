def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
    p_not_a = 1 - p_a
    p_b = (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a)

    p_a_given_b = (p_b_given_a * p_a) / p_b

    return p_a_given_b


p_a = 0.0002
p_b_given_a = 0.85
p_b_given_not_a = 0.05

print(bayes_theorem(p_a, p_b_given_a, p_b_given_not_a))