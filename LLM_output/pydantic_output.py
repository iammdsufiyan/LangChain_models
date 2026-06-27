from pydantic import BaseModel,EmailStr, Field
from typing import  Optional
class student(BaseModel):
    name:str = "md sufiyan"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10, default = 5)
new_student = {'age':25, 'email':'abc@gmail.com' , 'cgpa':9}

studentss = student(**new_student)

print(studentss)