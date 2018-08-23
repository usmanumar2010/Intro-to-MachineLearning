#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    # as we are predicting persons net worth so we are comparing net worth with predictions
    import numpy
    #hamary predictions of networth ko real networth say subtract kar rahay error find karnay kai leyai
    errors = net_worths-predictions
    #absolute ka function negative values ko b positive kar raha
    #percentile ka function
    threshold = numpy.percentile(numpy.absolute(errors), 90)

    cleaned_data = [(age, net_worth, error) for age, net_worth, error in zip(ages, net_worths, errors) if
                    abs(error) <= threshold]

    return cleaned_data

