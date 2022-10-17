class ironman():  
    def getResponse(self, args):
        """called when an AIML pattern from .aiml file matches and redirects responseController to run corresponding plugin
            args:
                args(string): string argument passed from AIML file to plugin script
            returns:
                a string of the next response to end user
        """
        return "Tony Stark is the Real Iron Man!!!"
