
def printSectionHeader(message):
    """
    Print a formatted section header.

    Args:
        message (str): The message to be displayed in the header.
    """
    print("\n" + "=" * 60)
    print(message.center(60))
    print("=" * 60 + "\n")

def printSectionFooter(message):
    """
    Print a formatted section footer.

    Args:
        message (str): The message to be displayed in the footer.
    """
    print('\n' + '=' * 60)
    print(message.center(60))
    print('=' * 60 + "\n")
