import dbm
from collections.abc import Iterable
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

class SimpleBagOfWords:
    """

    Parameters
    ----------

    handle_unknown : {'error', 'ignore', 'word_hashing'}, default='error'

    Attributes
    ----------
    vocab_size : vocal size

    Examples
    --------

    """
    def __init__(self, file: str = None, vocab_size: int = 0, handle_unknown='error'):
        self._db: dbm = dbm.open(file, 'c')
        self._handle_unknown: str = handle_unknown
        self._vocab_size: int = vocab_size
        self._encode: str = 'utf-8'

    def _validate_keywords(self):
        if self._handle_unknown not in ('error', 'ignore', 'word_hashing'):
            msg = ("handle_unknown should be either 'error' or 'ignore' or 'word_hashing', "
                   "got {0}.".format(self._handle_unknown))
            raise ValueError(msg)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def fit(self, raw_documents) -> None:
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """

        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
        elif not isinstance(raw_documents, Iterable):
            # not iterable
            raise TypeError(f"{type(raw_documents)} object is not iterable")
        else:
            # iterable
            for item in raw_documents:
                encoded = item.strip().encode(self._encode)
                if encoded not in self._db:
                    seq = len(self._db) + 1
                    self._db[encoded] = seq.to_bytes(10, byteorder='little')
                if self._vocab_size > 0 and self._vocab_size >= len(self._db):
                    break
        return self

    def transform(self, raw_documents) -> np.ndarray:
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """

        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")
        elif not isinstance(raw_documents, Iterable):
            # not iterable
            raise TypeError(f"{type(raw_documents)} object is not iterable")
        else:
            # iterable
            result = np.zeros(len(raw_documents), dtype='int32')
            for idx, item in enumerate(raw_documents):
                encoded = str(item).strip().encode(self._encode)
                if encoded in self._db:
                    result[idx] = int.from_bytes(self._db[encoded], byteorder='little')
                elif self._handle_unknown == 'error':
                    raise RuntimeError("_handle_unknown")
            return result

    def inverse_transform(self, X) -> np.ndarray:
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document-term matrix.

        Returns
        -------
        X_inv : list of arrays of shape (n_samples,)
            List of arrays of terms.
        """

        if isinstance(X, str):
            raise ValueError(
                "Iterable over int values expected, "
                "string object received.")
        elif not isinstance(X, Iterable):
            # not iterable
            raise TypeError(f"{type(X)}' object is not iterable")
        else:
            # iterable
            result = np.empty(len(X), dtype='object')
            encoded_values = [value.to_bytes(10, byteorder='little') for value in X]

            for key, item in self._db.items():
                if item in encoded_values:
                    result[encoded_values.index(item)] = key.decode(self._encode)
                elif self._handle_unknown == '':
                    raise RuntimeError("error")
            return result

    def clear(self) -> None:
        self._db.clear()
        return self

    def __len__(self) -> int:
        return len(self._db)

    def __contains__(self, key: str) -> bool:
        return self._db.__contains__(key)