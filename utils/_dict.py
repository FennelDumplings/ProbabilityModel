
class _Dict:
    """对内置字典的包装，用于表示分布"""

    def __init__(self, obj=None, label=None):
        """分布的初始化

        obj 可接受的类型:
            直方图: Hist
            概率质量函数：Pmf
            概率Cdf,
            Pdf,
            dict
            pandas Series
            list of pairs
        label: string label
        """
        self.label = label if label is not None else DEFAULT_LABEL
        self.d = {}

        # 分布是否经过了 log 变换
        self.log = False

        if obj is None:
            return

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.label = label if label is not None else obj.label

        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.d.update(obj.Items())
        elif isinstance(obj, pandas.Series):
            self.d.update(obj.value_counts().iteritems())
        else:
            # finally, treat it like a list
            self.d.update(Counter(obj))

        if len(self) > 0 and isinstance(self, Pmf):
            self.Normalize()

    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return '%s(%s)' % (cls, str(self.d))
        else:
            return self.label

    def __repr__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return '%s(%s)' % (cls, repr(self.d))
        else:
            return '%s(%s, %s)' % (cls, repr(self.d), repr(self.label))

    def __eq__(self, other):
        try:
            return self.d == other.d
        except AttributeError:
            return False

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        """Returns an iterator over keys."""
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def __getitem__(self, value):
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        self.d[value] = prob

    def __delitem__(self, value):
        del self.d[value]

    def Copy(self, label=None):
        """Returns a copy.

        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.

        label: string label for the new Hist

        returns: new _DictWrapper with the same type
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label
        return new

    def Scale(self, factor):
        """Multiplies the values by a factor.

        factor: what to multiply by

        Returns: new object
        """
        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new

    def Log(self, m=None):
        """Log transforms the probabilities.

        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        """Exponentiates the probabilities.

        m: how much to shift the ps before exponentiating

        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))

    def GetDict(self):
        """Gets the dictionary."""
        return self.d

    def SetDict(self, d):
        """Sets the dictionary."""
        self.d = d

    def Values(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def Items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def SortedItems(self):
        """Gets a sorted sequence of (value, freq/prob) pairs.

        It items are unsortable, the result is unsorted.
        """
        def isnan(x):
            try:
                return math.isnan(x)
            except TypeError:
                return False

        if any([isnan(x) for x in self.Values()]):
            msg = 'Keys contain NaN, may not sort correctly.'
            logging.warning(msg)

        try:
            return sorted(self.d.items())
        except TypeError:
            return self.d.items()

    def Render(self, **options):
        """Generates a sequence of points suitable for plotting.

        Note: options are ignored

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*self.SortedItems())

    def MakeCdf(self, label=None):
        """Makes a Cdf."""
        label = label if label is not None else self.label
        return Cdf(self, label=label)

    def Print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in self.SortedItems():
            print(val, prob)

    def Set(self, x, y=0):
        """Sets the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def Incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def Total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def MaxLike(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.d.values())

    def Largest(self, n=10):
        """Returns the largest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=True)[:n]

    def Smallest(self, n=10):
        """Returns the smallest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=False)[:n]
