import time

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='¨€'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    # Check if iteration is less than or equal to total
    if iteration > total:
        raise ValueError('Iteration value should be less than or equal to the total value')
    
    # Calculate percent complete
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    
    # Calculate filled length of bar
    filledLength = int(length * iteration // total)
    
    # Create progress bar string
    bar = fill * filledLength + '-' * (length - filledLength)
    
    # Print progress bar
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    
    # Print new line on complete
    if iteration == total: 
        print()
