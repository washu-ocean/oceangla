from builtins import property as _property, tuple as _tuple
from operator import itemgetter as _itemgetter
from collections import OrderedDict



class ScaledValueNode(tuple):
    """
    This is an extended `namedtuple` (from this post: https://stackoverflow.com/questions/17916853/how-named-tuples-are-implemented-internally-in-python), which type checks that the property `op` is a 2-tuple ()
    """
    __slots__ = ()
    _fields = ('op', 'varname')

    def __new__(_cls, op, varname):
        return _tuple.__new__(_cls, (op, varname))

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        result = new(cls, iterable)
        if len(result) != 2:
            raise TypeError('Expected 2 arguments, got %d' % len(result))
        return result

    def _replace(_self, **kwds):
        'Return a new Point object replacing specified fields with new values'
        result = _self._make(map(kwds.pop, ('op', 'varname'), _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + '(op=%r, varname=%r)' % self

    @property
    def __dict__(self):
        'A new OrderedDict mapping field names to their values'
        return OrderedDict(zip(self._fields, self))

    def _asdict(self):
        '''Return a new OrderedDict which maps field names to their values.
           This method is obsolete.  Use vars(nt) or nt.__dict__ instead.
        '''
        return self.__dict__

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    def __getstate__(self):
        'Exclude the OrderedDict from pickling'
        return None

    op = _property(_itemgetter(0), doc='Alias for field number 0')

    varname = _property(_itemgetter(1), doc='Alias for field number 1')

    sign, scalar = op


