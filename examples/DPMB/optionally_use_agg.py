#!python
def have_display():
    return 'DISPLAY' in os.environ

def optionally_use_agg():
    if not have_display():
        import matplotlib
        matplotlib.use('Agg')
