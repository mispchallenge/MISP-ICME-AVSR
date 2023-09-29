class Distancedict:

    def __init__(self,fixes):
        self.d_dict = {}
        self.fixes = fixes
        self.srcfixes= fixes[1:]
        self.tgtfix = fixes[0]
        for fix in  self.srcfixes:
            self.d_dict[fix] = 0.

    def __getitem__(self,key):
        return self.d_dict.get(key)

    def __setitem__(self,key,value):
        self.d_dict[key] = value

    def __delitem__(self,key):
        self.d_dict.pop(key)

    def setzero(self):
        for fix in self.srcfixes:
            self.d_dict[fix] = 0. 
    def checkzero(self):
        return all(list(self.d_dict.vaules()) == 0.) 

    def checkoom(self,fix): 
        fixindex = self.srcfixes.index(fix)
        if fixindex == 0:
            return True 
        else:
            if self.d_dict(fixindex-1) == 0.:
                self.setzero()
                return False
            else: 
                return True
  
               

         
