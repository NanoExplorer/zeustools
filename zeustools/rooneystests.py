if __name__ == "__main__":
    import doctest
    doctest.testfile("tests.txt")

    from zeustools import leapseconds
    leapseconds.run_tests(print_database=False)
    print("tests complete")