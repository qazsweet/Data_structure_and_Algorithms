# Iteration with loops

``for``loop
``enumerate``loop

* "in" keywords could be used to iterate over dictornay keys.

# Condidtional statements

``<``,``>``,``==``;
``if``, ``elif``, ``else``.

# dictornary 

def when_offered(courses, course):  
    # TODO: Fill out the function here.  
    semesters = []  
    for semester in courses:  
        if course in courses[semester]:  
            semesters.append(semester)  
    # TODO: Return list of semesters here.  
    return semesters  

