from tkinter import E


class PathNotFoundError(Exception):
    def  __init__(self,path,message,error=0):
        self.path=path
        self.message=message
        self.error_code=error
        super().__init__(self.message)

    def __str__(self):
        message=self.message+ " \n Path Defined : "+self.path +"\n Error code : "+ str(self.error_code)
        return message




class InvalidNumberofFrames(Exception):
    pass


class IndexOutOfBoundError(Exception):
    pass