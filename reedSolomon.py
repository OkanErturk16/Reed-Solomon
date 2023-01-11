import numpy as np
import utils
import numba

#TO DO Shortening yazilmali

@numba.jit(nopython = True)
def _convolve_GF(X,Y,gf_MUL_TABLE):
    Z = np.zeros(len(X) + len(Y) - 1, dtype = np.int64)
    for ii in range(len(X)):
        if X[ii] != 0:
            for jj in range(len(Y)):
                Z[ii+jj] ^= gf_MUL_TABLE[X[ii], Y[jj]]
    return Z

@numba.jit(nopython = True)
def _get_gf_poly_form(x,prim_poly):
    len_p = len(prim_poly)
    if len(x)<len_p:
        y  = np.zeros(len_p-1,dtype = np.uint64)
        y[-len(x):] = np.copy(x)
        return y
    else:
        for ii in range(len(x)-len_p+1):
            if x[ii] == 1:
                x[ii:(ii+len_p)] = np.logical_xor(x[ii:(ii+len_p)], prim_poly)
        return x[-(len_p-1):]

def _generate_MUL_table(q, prim_poly):
    gf_MUL_TABLE = np.zeros((q,q),dtype = np.uint64)
    m = int(np.log2(q))
    bin2dec_kernel = np.array(2**np.arange(m-1,-1,-1), dtype= np.int64)

    for ii in range(q):
        x1 = utils.d2b_rev(ii,m)
        for jj in range(q):
            x2                  = utils.d2b_rev(jj,m)
            conv_x1_x2          = np.convolve(x1, x2)%2
            gf_MUL_TABLE[ii,jj] = int(np.sum(_get_gf_poly_form(conv_x1_x2,prim_poly)*bin2dec_kernel))
    gf_MUL_TABLE = np.array(gf_MUL_TABLE, dtype = np.int64)
    return gf_MUL_TABLE

def _generate_DIV_table(q, gf_MUL_TABLE):
    gf_DIV_TABLE   = np.zeros((q,q),dtype = np.uint64)
    for ii in range(1,q):
        for jj in range(1,q):
            gf_MUL_ii_jj = gf_MUL_TABLE[ii,jj]
            gf_DIV_TABLE[gf_MUL_ii_jj,jj] = ii
    gf_DIV_TABLE = np.array(gf_DIV_TABLE, dtype = np.int64)
    return gf_DIV_TABLE

def _generate_prim_POWS(q, gf_MUL_TABLE,prim_elem):
    prim_POWS = np.zeros((q),dtype=np.int64)
    prim_POWS[0] = 1
    for ii in range(1,q):
            prim_POWS[ii] = gf_MUL_TABLE[prim_POWS[ii-1],prim_elem]
    return prim_POWS

def _generate_prim_POW_TABLE(q,n_k,prim_elem,gf_MUL_TABLE,prim_POWS,b):
    prim_POW_TABLE = np.zeros((n_k,q-1),dtype=np.int64)
    prim_POW_TABLE[:,-1] = 1
    for jj in range(n_k):
        alpha_scale = prim_POWS[b]
        for _ in range(jj):
            alpha_scale = gf_MUL_TABLE[alpha_scale,prim_elem]
        for ii in range(1,q-1):
            prim_POW_TABLE[jj,q-1-ii-1] = gf_MUL_TABLE[prim_POW_TABLE[jj,q-1-ii],alpha_scale]
    return prim_POW_TABLE

def _generate_generator_polynomial(n_k, gf_MUL_TABLE,prim_POWS, prim_elem, b):
    """
        return : g_X 'np.array'
        g_X[0], MSB olarak tanimli
        Ornek ``g_X = [1,0,0,1,1]`` ise
        g(X) = X^4 + X + 1 
    """
    # "generator polynomial" carpanlarini "g_factor" listesinde olustur
    g_factors = []
    alpha_init = prim_POWS[b]
    for ii in range(n_k):
        g_factors.append(np.array([1,alpha_init], dtype = np.int64))
        alpha_init = gf_MUL_TABLE[alpha_init,prim_elem]

    # "g_factors" listesindeki carpanları konvole ederek "g_X" generator polynomial'i olustur
    g_X     = np.zeros(len(g_factors)+1,dtype=np.int64)
    g_X[-1] = 1 #baslangic degeri
    for ii in range(len(g_factors)):
        g_iter = 0*g_X
        for jj in range(len(g_X)-1, 0, -1):
            g_factors_scale    = 1*g_factors[ii]
            g_factors_scale[1] = gf_MUL_TABLE[g_X[jj], g_factors[ii][1]]
            g_factors_scale[0] = gf_MUL_TABLE[g_X[jj], g_factors[ii][0]]
            g_iter[jj]         = g_factors_scale[1] ^ g_iter[jj]
            g_iter[jj-1]       = g_factors_scale[0] ^ g_iter[jj-1]
        g_X = np.copy(g_iter)
    return g_X

@numba.jit(nopython = True)
def _RS_encoder(u,g_X,q,g_X_MUL):
    v = np.zeros(q-1, dtype= np.int64)
    v[:len(u)] = u
    len_g = len(g_X)
    for ii in range(q-len_g):
        if v[ii] == 0:
            continue
        else:
            for jj in range(1,len_g):
                v[ii+jj] ^= g_X_MUL[v[ii],jj]
            v[ii] = 0
    v[:len(u)] = u
    return v

@numba.jit(nopython = True)
def _syndrome(r,n_k,gf_MUL_TABLE,prim_POW_TABLE):
    s_X = np.zeros(n_k, dtype = np.int64)
    for ii in range(len(r)):
        if r[ii] != 0:
            for jj in range(n_k):
                s_X[jj] ^= gf_MUL_TABLE[prim_POW_TABLE[jj,ii],r[ii]]
    return s_X

@numba.jit(nopython = True)
def _BerlekampMassey(s_X, t, gf_MUL_TABLE,gf_DIV_TABLE):
    sigma_X    = np.zeros(2*t,dtype = np.int64)
    sigma_X[0] = 1
    B_X = np.zeros(2*t,dtype = np.int64)
    B_X[0] = 1

    sigma_X_new = np.zeros(2*t,dtype = np.int64)

    L     = 0
    delta = 0
    for jj in range(1,2*t+1):
        S_jj_tilde = 0

        for ii in range(1,L+1):
            S_jj_tilde ^= gf_MUL_TABLE[sigma_X[ii],s_X[jj-ii-1]]
        Delta_jj = s_X[jj-1] ^ S_jj_tilde
        if Delta_jj != 0 and (2*L)<=(jj-1):
            delta = 1
        else:
            delta = 0

        B_shift = np.roll(B_X,1)
        B_shift[0] = 0

        sigma_X_new = np.copy(sigma_X)
        B_shift_MUL_Delta_jj = gf_MUL_TABLE[B_shift,Delta_jj]

        sigma_X_new ^= B_shift_MUL_Delta_jj

        if delta:
            Delta_jj_inv = gf_DIV_TABLE[1,Delta_jj]
            B_X = gf_MUL_TABLE[sigma_X,Delta_jj_inv]
            L = jj-L
        else:
            B_X = np.copy(B_shift)
        sigma_X = np.copy(sigma_X_new)

    return sigma_X

@numba.jit(nopython = True)
def _ChienSearch(sigma_X,t,prim_POWS,gf_MUL_TABLE,gf_DIV_TABLE):
    locations = np.zeros(t,dtype = np.int64)
    vv  = 0 #initial root number
    jj  = 0
    while vv<t and jj<(len(prim_POWS)-1):
        parity = 0
        root_init = gf_DIV_TABLE[1,prim_POWS[jj]]
        root = 1
        for ii in range(len(sigma_X)):
            parity ^= gf_MUL_TABLE[sigma_X[ii],root]
            root    = gf_MUL_TABLE[root,root_init]
        if parity == 0:
            locations[vv] = jj
            vv += 1
        jj +=1
    locations = locations[:vv]
    return locations

@numba.jit(nopython = True)
def _ForneyAlgorithm(s_X, sigma_X, locations, gf_MUL_TABLE, gf_DIV_TABLE, prim_POWS, q, t, b):
    Omega_X = np.zeros(len(s_X) + len(sigma_X) -1, dtype= np.int64)
    Lmbda_derivative  = np.copy(sigma_X)
    Lmbda_derivative[0::2] = 0
    Lmbda_derivative = np.roll(Lmbda_derivative, -1)
    Omega_X = _convolve_GF(s_X, sigma_X, gf_MUL_TABLE)
    Omega_X = Omega_X[:(2*t)]
    error   = np.zeros(len(locations), dtype = np.int64)
    for ii in range(len(locations)):
        root_ii        = prim_POWS[locations[ii]]
        #%% TO DO BURAYI ROOT DERECESI UZERINDEN YAZARSAK DAHA KOLAY OLACAK !
        root_ii_inv = gf_DIV_TABLE[1,root_ii]

        X_j_iter                  = 1
        lmbda_derivative_root_inv = 0

        for jj in range(len(Lmbda_derivative)):
            lmbda_derivative_root_inv ^= gf_MUL_TABLE[Lmbda_derivative[jj],X_j_iter]
            X_j_iter = gf_MUL_TABLE[X_j_iter,root_ii_inv]
        X_j_iter       = 1
        Omega_root_inv = 0
        for jj in range(len(Omega_X)):
            Omega_root_inv ^= gf_MUL_TABLE[Omega_X[jj],X_j_iter]
            X_j_iter = gf_MUL_TABLE[X_j_iter,root_ii_inv]

        error[ii] = prim_POWS[((locations[ii]*(1-b))%(q-1))]
        error[ii] = gf_MUL_TABLE[error[ii],Omega_root_inv]
        error[ii] = gf_DIV_TABLE[error[ii],lmbda_derivative_root_inv] 
    return error

@numba.jit(nopython = True)
def _correctErrors(r,q,locations,error):
    v_hat = np.copy(r)
    v_hat[q-2-locations] ^= error
    return v_hat

@numba.jit(nopython = True)
def _decode_RS_to_RS(r,n_k, q, t, b,
                    gf_MUL_TABLE,  gf_DIV_TABLE,
                    prim_POWS,     prim_POW_TABLE):

    s_X = _syndrome(r,
                    n_k,
                    gf_MUL_TABLE,
                    prim_POW_TABLE)

    if np.any(s_X):
        sigma_X = _BerlekampMassey(s_X,
                                    t,
                                    gf_MUL_TABLE,
                                    gf_DIV_TABLE)
        
        locations = _ChienSearch(sigma_X,
                                t,
                                prim_POWS,
                                gf_MUL_TABLE,
                                gf_DIV_TABLE)

        error = _ForneyAlgorithm(s_X,
                                sigma_X,
                                locations, 
                                gf_MUL_TABLE, 
                                gf_DIV_TABLE, 
                                prim_POWS, 
                                q,
                                t,
                                b)
        
        v_hat = _correctErrors(r, q, locations, error)
        return v_hat
    else:
        return np.copy(r)

@numba.jit(nopython = True)
def _decode_RS_to_RS_shortened(r_RS,n_k, q,n, t, b,
                    gf_MUL_TABLE,  gf_DIV_TABLE,
                    prim_POWS,     prim_POW_TABLE):
    r_RS_appended = np.concatenate((np.zeros(q -1 - n, dtype=np.int64), r_RS))
    v_RS = _decode_RS_to_RS(r_RS_appended,n_k, q, t, b,
                            gf_MUL_TABLE,  gf_DIV_TABLE,
                            prim_POWS,     prim_POW_TABLE)
    return v_RS[-n:]


#------------------------------------------------------------
#ENCODER
class ReedSolomonEncoder:
    """
        ## Reed Solomon Kodlayici

        *q*: GF(q) sonlu alani icin "q" degeri

        *n*: RS simge cinsinden kodlayici cikis uzunlugu

        *k*: RS simge cinsinden kodlayici giris uzunlugu  

        *prim_poly*: GF(q)'nun tanimalanmasinda kullanilacak olan "primitive polynomial". prim_poly[0] en yuksek dereceli elemani. Ornek olarak p(X) = X^8 + X^5 + X^3 + X + 1 primitif polinomu, numpy array cinsinden su bicimde tanimlanir:
        >>> prim_poly = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1], dtype = np.int64)

        *prim_elem*: GF(q) icin kullanilan primitif eleman. (Tipik olarak 2 secilebilir.)

        *b*:  Uretec polinomunun en dusuk dereceli primitif elemaninin derecesi.
        "*b*" degerinin kullanimina ornek: Kodlayici asagidaki uretec polinomunu olusturacak:        *g(X) = (X+alpha^(b))(X + alpha^(b+1))...(X + alpha^(b+2*t))*
        Burada "alpha" ile primitiv eleman ve "t" ile duzeltilebilir RS sembol hatasi sayisi gosterilmektedir.
    """

    def __init__(self,q,n,k,prim_poly,prim_elem=2,b=1):
        self.q    = q
        self.n    = n
        self.k    = k
        self.num_bits_for_RS_symbol = int(np.log2(q))

        self.is_shortened  = False if ((self.q-1)==self.n) else True

        self._n_k = self.n - self.k
        self.t    = int(self._n_k/2)
        self.b    = b
        self.prim_poly = prim_poly
        self.prim_elem = prim_elem
        
        self._gf_MUL_TABLE = _generate_MUL_table(self.q, self.prim_poly)
        self._gf_DIV_TABLE = _generate_DIV_table(self.q, self._gf_MUL_TABLE)

        self._prim_POWS = np.zeros((self.q), dtype = np.int64)
        self._prim_POWS[0] = 1
        for ii in range(1,self.q):
                self._prim_POWS[ii] = self._gf_MUL_TABLE[self._prim_POWS[ii-1],self.prim_elem]

        self.g_X = _generate_generator_polynomial(self._n_k,
                                                    self._gf_MUL_TABLE,
                                                    self._prim_POWS,
                                                    self.prim_elem,
                                                    self.b)

        self._g_X_MUL = np.zeros(shape = (self.q, len(self.g_X)), dtype = np.int64)
        for ii in range(self.q):
            for jj in range(len(self.g_X)):
                self._g_X_MUL[ii,jj] = self._gf_MUL_TABLE[ii, self.g_X[jj]]



    def encode_RS_to_RS(self,u_RS):
        """
            # RS sembol kodlayici

            Giris ve cikis elemanlari RS sembolleri cinsinden olan kodlama metodu

            ### Parametreler:

            *u_RS*: RS simgeleri cinsinden bilgi dizisi ` np.array `. Veri tipi ` np.int64 ` olmalidir. 

            ### Return

            v_RS: RS simgeleri cinsinden kodsozcugu ` np.array `. Veri tipi ` np.int64 ` olmalidir. 
        """
        if self.is_shortened:
            u_RS_appended = np.concatenate((np.zeros(self.q - 1 - self.n), u_RS))
            v_RS = _RS_encoder(u_RS_appended,
                                self.g_X,
                                self.q,
                                self._g_X_MUL)
            return v_RS[-self.n:]

        else:
            return _RS_encoder(u_RS,
                                self.g_X,
                                self.q,
                                self._g_X_MUL)

    def encode_bit_to_RS(self,u_bit,order = 'MSB_first'):
        if order == 'MSB_first':
            bit2dec_kernel = 2**np.arange(self.num_bits_for_RS_symbol-1,-1,-1)
        if order == 'LSB_first':
            bit2dec_kernel = 2**np.arange(0,self.num_bits_for_RS_symbol)
        
        u_bit_reshape = np.reshape(u_bit, (int(len(u_bit)/self.num_bits_for_RS_symbol),self.num_bits_for_RS_symbol))
        u_RS = np.sum(u_bit_reshape*bit2dec_kernel, axis=1)
        return self.encode_RS_to_RS(u_RS)

    def encode_bit_to_bit(self,u_bit, order = 'MSB_first'):
        if order == 'MSB_first':
            u_RS  = self.encode_bit_to_RS(u_bit, order = 'MSB_first')
            u_bit = utils.d2b_rev_array(u_RS, self.num_bits_for_RS_symbol)
            return u_bit.reshape(-1)
        if order == 'LSB_first':
            u_RS  = self.encode_bit_to_RS(u_bit, order = 'LSB_first')
            u_bit = utils.d2b_rev_array(u_RS, self.num_bits_for_RS_symbol)
            return u_bit.reshape(-1)
#-----------------------------------------------------------
#DECODER
class ReedSolomonDecoder:
    """
        ## Reed Solomon Kodcozucu

        *q*: GF(q) sonlu alani icin "q" degeri

        *n*: RS simge cinsinden kodlayici cikis uzunlugu

        *k*: RS simge cinsinden kodlayici giris uzunlugu  

        *prim_poly*: GF(q)'nun tanimalanmasinda kullanilacak olan "primitive polynomial". prim_poly[0] en yuksek dereceli elemani. Ornek olarak p(X) = X^8 + X^5 + X^3 + X + 1 primitif polinomu, numpy array cinsinden su bicimde tanimlanir:
        >>> prim_poly = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1], dtype = np.int64)

        *prim_elem*: GF(q) icin kullanilan primitif eleman. (Tipik olarak 2 secilebilir.)

        *b*:  Uretec polinomunun en dusuk dereceli primitif elemaninin derecesi.
        "*b*" degerinin kullanimina ornek: Kodlayici asagidaki uretec polinomunu olusturacak:        *g(X) = (X+alpha^(b))(X + alpha^(b+1))...(X + alpha^(b+2*t))*
        Burada "alpha" ile primitiv eleman ve "t" ile duzeltilebilir RS sembol hatasi sayisi gosterilmektedir.
    """
    def __init__(self,q,n,k,prim_poly,prim_elem=2,b=1):
        self.q    = q
        self.n    = n
        self.k    = k
        self.num_bits_for_RS_symbol = int(np.log2(q))

        self.is_shortened  = False if ((self.q-1)==self.n) else True


        self._n_k = self.n - self.k
        self.t    = int(self._n_k/2)
        self.b    = b
        self.prim_poly = prim_poly
        self.prim_elem = prim_elem
        
        self._gf_MUL_TABLE = _generate_MUL_table(self.q, self.prim_poly)
        self._gf_DIV_TABLE = _generate_DIV_table(self.q, self._gf_MUL_TABLE)

        self._prim_POWS = np.zeros((self.q), dtype = np.int64)
        self._prim_POWS[0] = 1
        for ii in range(1,self.q):
                self._prim_POWS[ii] = self._gf_MUL_TABLE[self._prim_POWS[ii-1],self.prim_elem]

        self.g_X = _generate_generator_polynomial(self._n_k,
                                                    self._gf_MUL_TABLE,
                                                    self._prim_POWS,
                                                    self.prim_elem,
                                                    self.b)

        self._g_X_MUL = np.zeros(shape = (self.q, len(self.g_X)), dtype = np.int64)
        for ii in range(self.q):
            for jj in range(len(self.g_X)):
                self._g_X_MUL[ii,jj] = self._gf_MUL_TABLE[ii, self.g_X[jj]]


        self._prim_POW_TABLE = np.zeros((self._n_k, self.q - 1),dtype=np.int64)
        self._prim_POW_TABLE[:,-1] = 1
        for jj in range(self._n_k):
            alpha_scale = self._prim_POWS[b]
            for _ in range(jj):
                alpha_scale = self._gf_MUL_TABLE[alpha_scale,self.prim_elem]
            for ii in range(1,q-1):
                self._prim_POW_TABLE[jj,self.q-1-ii-1] = self._gf_MUL_TABLE[self._prim_POW_TABLE[jj,q-1-ii],alpha_scale]


    def decode_RS_to_RS(self,r_RS):
        if self. is_shortened:
            return _decode_RS_to_RS_shortened(r_RS,
                                                self._n_k,
                                                self.q,
                                                self.n,
                                                self.t,
                                                self.b,
                                                self._gf_MUL_TABLE,
                                                self._gf_DIV_TABLE,
                                                self._prim_POWS,
                                                self._prim_POW_TABLE)
        else:
            return _decode_RS_to_RS(r_RS,
                                    self._n_k,
                                    self.q,
                                    self.t,
                                    self.b,
                                    self._gf_MUL_TABLE,
                                    self._gf_DIV_TABLE,
                                    self._prim_POWS,
                                    self._prim_POW_TABLE)

    def decode_bit_to_RS(self,v_bit,order = 'MSB_first'):
        if order == 'MSB_first':
            bit2dec_kernel = 2**np.arange(self.num_bits_for_RS_symbol-1,-1,-1)
            v_bit_reshape = np.reshape(v_bit, (int(len(v_bit)/self.num_bits_for_RS_symbol),self.num_bits_for_RS_symbol))
            v_RS = np.array(np.sum(v_bit_reshape*bit2dec_kernel, axis=1),dtype=np.int64)
            return self.decode_RS_to_RS(v_RS)
        if order == 'LSB_first':
            bit2dec_kernel = 2**np.arange(0,self.num_bits_for_RS_symbol)
            print('IMPLEMENTE EDILECEK!')

    def decode_bit_to_bit(self,u_bit, order = 'MSB_first'):
        if order == 'MSB_first':
            u_RS  = self.decode_bit_to_RS(u_bit, order = 'MSB_first')
            u_bit = utils.d2b_rev_array(u_RS, self.num_bits_for_RS_symbol)
            return u_bit.reshape(-1)
        if order == 'LSB_first':
            print('HENUZ YAZILMADI!')
#--------------------------------------------------------------------------------------
