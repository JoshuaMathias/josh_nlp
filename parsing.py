
def binToUTF8(string):
	return ('%x' % int(string, 2)).decode('hex').decode('utf-8')

