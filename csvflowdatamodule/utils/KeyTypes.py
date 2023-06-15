import re

def parse_request(request, authorised_colnames=None) :
    """
    We can build a request from a column name and a signed like Flow-4 that indicates
    which flow we want to retrieve with respect to a certain index.
    Here we desinbiguate the name from the gap from the request
    Args :
        request (str) : string with '{ColName}{opt:+/-}{opt:offset}'
        authorised_colnames ([str]) : list of colnames authorised. If None no constraint
    Return :
        colname (str) : string with the name of the column
        gap (signed int) : offset from the reference index
    """
    rec = re.compile("([A-z]+)([+-])?([0-9]+)?$")
    match = rec.match(request)

    if match is not None :
        _, colname, sign, gap = [match[i] for i in range(4)]
    elif match is None :
        rec = re.compile("([A-z]+)([+-])?([0-9]+)?(Path)$")
        match = rec.match(request)
        colname, sign, gap = match[1] + match[4], match[2], match[3]
    assert match is not None, f'Problem with request : {request}'

    if authorised_colnames is not None :
        assert colname  in authorised_colnames, f'Request {colname} no in {authorised_colnames}'

    if sign is None or gap is None :
        return colname, 0
    else :
        return colname, int(str(sign+gap))


def extract_ascending_list(sequence_request, colname) :
    """
    Return a list of request in ascending order in term of gap and filter
    to only return request with given colname.
    Args :
        sequence_request ([str]) : List of request to process.
                                   ex : ['Flow-2', 'Flow+4', 'Flow', 'Image-4', 'Flow-1', 'GtMask']
        colname (str) : colname to select
                        ex : 'Flow'
    Returns
        asc_seq_request ([str]) : with the selected colname. empty sequence if None
                        ex : ['Flow-2', 'Flow-1', 'Flow', 'Flow+4']
    """
    fl =  list(filter(lambda r: parse_request(r)[0] == colname, sequence_request))
    return sorted(fl, key=lambda r: parse_request(r)[1])
