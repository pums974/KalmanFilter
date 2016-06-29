subroutine derx_DF2_f(field, derx, mat, n, m)
    implicit none
  integer,intent(in) :: n, m
  double precision,intent(in) :: field(n,m)
  double precision,intent(in) :: mat(n,m,3)
  double precision,intent(out) :: derx(n,m)
  integer :: start,end, i,j,k

  derx = 0.
  do j = 1,m
    do i = 1,n
      start = max(i - 1, 1)
      end = min(start + 2, n)
      start = end - 2
      do k = start,end
            derx(i,j) = derx(i,j) + field(k,j) * mat(i,j,k - start+1)
       enddo
     enddo
   enddo
end subroutine derx_DF2_f

subroutine dery_DF2_f(field, dery, mat, n, m)
    implicit none
  integer,intent(in) :: n, m
  double precision,intent(in) :: field(n,m)
  double precision,intent(in) :: mat(n,m,3)
  double precision,intent(out) :: dery(n,m)
  integer :: start,end, i,j,k

  dery = 0.
  do j = 1,m
      start = max(j - 1, 1)
      end = min(start + 2, m)
      start = end - 2
      do k = start,end
        do i = 1,n
            dery(i,j) = dery(i,j) + field(i,k) * mat(i,j,k - start+1)
       enddo
     enddo
   enddo
end subroutine dery_DF2_f


subroutine derx_upwind_f(field, derx, velofield,dx, n, m)
    implicit none
  integer,intent(in) :: n, m
  double precision,intent(in) :: field(n,m),dx
  double precision,intent(in) :: velofield(n,m,2)
  double precision,intent(out) :: derx(n,m)
  integer :: i,j,k
  double precision :: ap,am,rdx
  integer,parameter :: order=2

!  rdx = 1./dx
!  derx = 0.
!  do j = 1,m
!    do i = 1,n
!        k=0
!        if (i > 1) then
!            ap = max(velofield(i,j,1) + velofield(i - 1,j,1), 0.d0)*0.5d0
!            derx(i,j) = derx(i,j) + ap * (field(i,j) - field(i-1,j)) * rdx
!            k=k+1
!        endif
!        if (i < n) then
!            am = min(velofield(i,j,1) + velofield(i + 1,j,1), 0.d0)*0.5d0
!            derx(i,j) = derx(i,j) + am * (field(i+1,j) - field(i,j)) * rdx
!            k=k+1
!        endif
!!        if (k==2) then
!!          if (ap - am < 1e-1) then
!!            derx(i,j) = velofield(i,j,1) * (field(i+1,j) - field(i-1,j)) * rdx * 0.5
!!            print*,"coucou",ap - am
!!            k=1
!!          endif
!!        endif
!        derx(i,j)=derx(i,j)/k
!     enddo
!   enddo
  derx = 0.
if (order==2) then
  rdx = 0.5/dx
  do j = 1,m
    do i = 3,n-2
        ap = max(velofield(i,j,1) + velofield(i - 1,j,1), 0.d0)*0.25d0
        am = min(velofield(i,j,1) + velofield(i + 1,j,1), 0.d0)*0.25d0
        derx(i,j) = derx(i,j) + ap * (3.*field(i,j) - 4.*field(i-1,j)+field(i-2,j)) * rdx
        derx(i,j) = derx(i,j) + am * (-field(i+2,j) + 4.*field(i+1,j) - 3.*field(i,j)) * rdx
    enddo
  enddo
  rdx = 1./dx
  do j = 1,m
     i = 2
        ap = max(velofield(i,j,1) + velofield(i - 1,j,1), 0.d0)*0.25d0
        am = min(velofield(i,j,1) + velofield(i + 1,j,1), 0.d0)*0.25d0
        derx(i,j) = derx(i,j) + ap * (field(i,j) - field(i-1,j)) * rdx
        derx(i,j) = derx(i,j) + am * (field(i+1,j) - field(i,j)) * rdx
     i = n-1
        ap = max(velofield(i,j,1) + velofield(i - 1,j,1), 0.d0)*0.25d0
        am = min(velofield(i,j,1) + velofield(i + 1,j,1), 0.d0)*0.25d0
        derx(i,j) = derx(i,j) + ap * (field(i,j) - field(i-1,j)) * rdx
        derx(i,j) = derx(i,j) + am * (field(i+1,j) - field(i,j)) * rdx
  enddo
else
  rdx = 1./dx
  do j = 1,m
    do i = 2,n-1
        ap = max(velofield(i,j,1) + velofield(i - 1,j,1), 0.d0)*0.25d0
        am = min(velofield(i,j,1) + velofield(i + 1,j,1), 0.d0)*0.25d0
        derx(i,j) = derx(i,j) + ap * (field(i,j) - field(i-1,j)) * rdx
        derx(i,j) = derx(i,j) + am * (field(i+1,j) - field(i,j)) * rdx
    enddo
  enddo
endif
  do j = 1,m
     i = 1
     am = min(velofield(i,j,1) + velofield(i + 1,j,1), 0.d0)*0.5d0
     derx(i,j) = am * (field(i+1,j) - field(i,j)) * rdx
  enddo
  do j = 1,m
     i = n
     ap = max(velofield(i,j,1) + velofield(i - 1,j,1), 0.d0)*0.5d0
     derx(i,j) = ap * (field(i,j) - field(i-1,j)) * rdx
  enddo

end subroutine derx_upwind_f

subroutine dery_upwind_f(field, dery, velofield,dy, n, m)
    implicit none
  integer,intent(in) :: n, m
  double precision,intent(in) :: field(n,m),dy
  double precision,intent(in) :: velofield(n,m,2)
  double precision,intent(out) :: dery(n,m)
  integer :: i,j
  double precision :: ap,am,rdy
  integer,parameter :: order = 2
!  rdy = 1./dy
!  dery = 0.
!  do j = 1,m
!    do i = 1,n
!        ap = max(velofield(i,j,2), 0.d0)*0.5
!        am = min(velofield(i,j,2), 0.d0)*0.5
!        if (j > 1) &
!            dery(i,j) = dery(i,j) + ap * (field(i,j) - field(i,j-1)) * rdy
!        if (j < n) &
!            dery(i,j) = dery(i,j) + am * (field(i,j+1) - field(i,j)) * rdy
!     enddo
!   enddo
   
   
  dery = 0.
if (order==2) then
  rdy = 0.5/dy
  do j = 3,m-2
    do i = 1,n
        ap = max(velofield(i,j,2) + velofield(i,j - 1,2), 0.d0)*0.25d0
        am = min(velofield(i,j,2) + velofield(i,j + 1,2), 0.d0)*0.25d0
        dery(i,j) = dery(i,j) + ap * (3.*field(i,j) - 4.*field(i,j-1)+field(i,j-2)) * rdy
        dery(i,j) = dery(i,j) + am * (-field(i,j+2) + 4.*field(i,j+1) - 3.*field(i,j)) * rdy
    enddo
  enddo
  rdy = 1./dy
  j = 2
  do i = 1,n
        ap = max(velofield(i,j,2) + velofield(i,j - 1,2), 0.d0)*0.25d0
        am = min(velofield(i,j,2) + velofield(i,j + 1,2), 0.d0)*0.25d0
        dery(i,j) = dery(i,j) + ap * (field(i,j) - field(i,j-1)) * rdy
        dery(i,j) = dery(i,j) + am * (field(i,j+1) - field(i,j)) * rdy
  enddo
  j = m-1
  do i = 1,n
        ap = max(velofield(i,j,2) + velofield(i,j - 1,2), 0.d0)*0.25d0
        am = min(velofield(i,j,2) + velofield(i,j + 1,2), 0.d0)*0.25d0
        dery(i,j) = dery(i,j) + ap * (field(i,j) - field(i,j-1)) * rdy
        dery(i,j) = dery(i,j) + am * (field(i,j+1) - field(i,j)) * rdy
  enddo
else
  rdy = 1./dy
  do j = 2,m-1
    do i = 1,n
        ap = max(velofield(i,j,2) + velofield(i,j - 1,2), 0.d0)*0.25d0
        am = min(velofield(i,j,2) + velofield(i,j + 1,2), 0.d0)*0.25d0
        dery(i,j) = dery(i,j) + ap * (field(i,j) - field(i,j-1)) * rdy
        dery(i,j) = dery(i,j) + am * (field(i,j+1) - field(i,j)) * rdy
     enddo
   enddo
endif
   j=1
   do i = 1,n
      am = min(velofield(i,j,2) + velofield(i,j + 1,2), 0.d0)*0.5d0
      dery(i,j) = am * (field(i,j+1) - field(i,j)) * rdy
   enddo
   j = m
   do i = 1,n
      ap = max(velofield(i,j,2) + velofield(i,j - 1,2), 0.d0)*0.5d0
      dery(i,j) = ap * (field(i,j) - field(i,j-1)) * rdy
   enddo
   
   




   
end subroutine dery_upwind_f

subroutine kalman_apply_f(Phi, S1, Q, M, R, Y, X1,s,x,n1,n2)
  implicit none
  integer,intent(in) :: n1,n2
  double precision,intent(in) :: Phi(n1,n1), M(n2,n1), R(n2,n2), Q(n1,n1), Y(n2,1)
  double precision ,intent(in):: S1(n1,n1),X1(n1,1)
  double precision,intent(out):: S(n1,n1),X(n1,1)

  double precision,allocatable :: innovation_covariance(:,:) , innovation(:,:), Id(:,:), k(:,:)
  double precision,allocatable :: innovation_covariance0(:,:), innovation0(:,:), S0(:,:), k0(:,:), x0(:,:)
  integer :: i,INFO
    allocate(innovation_covariance(n2,n2) , innovation(n2,1), Id(n1,n1), k(n1,n2))
    allocate(innovation_covariance0(n2,n2), innovation0(n2,1), S0(n1,n1), k0(n1,n2), x0(n1,1))
!      Id=0.
!      do i = 1,n1
!        Id(i,i)=1.
!      enddo
      
      ! -------------------------Prediction step-----------------------------
!      S = matmul(Phi,matmul(S1,transpose(Phi))) + Q
      S = transpose(Phi)
      S0  = matmul(S1,S)
      S = matmul(Phi,S0)
      S0 = S + Q

      ! ------------------------Observation step-----------------------------
!      innovation_covariance = matmul(M,matmul(S,transpose(M))) + R
      k0 = transpose(M)
      K = matmul(S0,k0)
      innovation_covariance0 = matmul(M,k)
      innovation_covariance = innovation_covariance0 + R

!      innovation = Y - matmul(M,X1)
      innovation0 = matmul(M,X1)
      innovation = Y - innovation0


      ! ---------------------------Update step-------------------------------
      innovation_covariance0 = inv(innovation_covariance)
!      k = matmul(s,matmul(transpose(m),innovation_covariance))
      k = transpose(M)
      k0 = matmul(k,innovation_covariance0)
      k = matmul(s0,k0)
      
!      x = x1 + matmul(k,innovation)
      x0 = matmul(k,innovation)
      x = x1 + x0
      
!      s = matmul(Id - matmul(k,m),s)
      Id = - matmul(k,m)
      do i = 1,n1
        Id(i,i)=1. + Id(i,i)
      enddo
      s = matmul(Id,s0)
            

      deallocate(innovation_covariance, innovation, Id, k)
      deallocate(innovation_covariance0, innovation0, S0, k0, x0)
    
contains


!subroutine kalman_full_lapack(mu, Sigma, H, INFO, R, Sigmavar_2, data, muvar_2, k, n)
!implicit none

!integer, intent(in) :: k
!integer, intent(in) :: n
!double precision, intent(in) :: Sigma(n, n)        !  Sigma
!double precision, intent(in) :: H(k, n)            !  H
!double precision, intent(in) :: mu(n,1)              !  mu
!double precision, intent(in) :: R(k, k)            !  R, H*Sigma*H' + R
!double precision, intent(in) :: data(k,1)            !  (H*Sigma*H' + R)^-1*((-1)*data + H*mu), data, (-1)*   data + H*mu
!integer, intent(out) :: INFO             !  INFO
!double precision, intent(out) :: muvar_2(n,1)        !  mu, Sigma*H'*(H*Sigma*H' + R)^-1*((-1)*data + H*  mu) + mu
!double precision, intent(out) :: Sigmavar_2(n, n)  !  Sigma, (-1)*Sigma*H'*(H*Sigma*H' + R)^-1*H* Sigma + Sigma
!double precision :: var_17(n, k)                   !  Sigma*H', 0
!double precision :: Hvar_2(k, n)                   !  (H*Sigma*H' + R)^-1*H, H
!double precision :: var_11(n,1)                      !  0, H'*(H*Sigma*H' + R)^-1*((-1)*data + H*mu)
!double precision :: var_19(n, n)                   !  0, H'*(H*Sigma*H' + R)^-1*H
!double precision :: var_20(n, n)                   !  H'*(H*Sigma*H' + R)^-1*H*Sigma, 0

!double precision :: R1(k, k)
!double precision :: R2(k, k)
!double precision :: data1(k,1)
!double precision :: X1(n,1)

!call dcopy(k**2, R, 1, R1, 1)
!call dcopy(n, mu, 1, muvar_2, 1)
!call dcopy(n**2, Sigma, 1, Sigmavar_2, 1)
!call dcopy(k*n, H, 1, Hvar_2, 1)

!!call dsymm('L', 'U', n, k, 1.d0, Sigma, n, H, k, 0.d0, var_17, n)
!var_17 = matmul(Sigma,transpose(H))
!R1 = matmul(H,var_17) + R1
!call dcopy(k**2, R1, 1, R2, 1)

!data1 = data - matmul(H,mu)

!call dposv('U', k, n, R1, k, Hvar_2, k, INFO)
!call dposv('U', k, 1, R2, k, data1, k, INFO)                
!var_19 = matmul(transpose(H),Hvar_2)
!var_11 = matmul(transpose(H),data1)

!!call dsymm('L', 'U', n, n, 1.d0, var_19, n, Sigma, n, 0.d0, var_20, n)
!!call dsymm('L', 'U', n, 1, 1.d0, Sigma, n, var_11, n, 1.d0, muvar_2, n)
!var_20 = matmul(var_19,Sigma)
!muvar_2 = muvar_2 + matmul(Sigma,var_11)
!Sigmavar_2 = Sigmavar_2 - matmul(Sigmavar_2,var_20)


!RETURN
!END subroutine kalman_full_lapack


! Wrapper around dgemm
function matmul(A,B,C,ta,tb) result(AB)
implicit none
  double precision, dimension(:,:), intent(in) :: A,B
  double precision, dimension(:,:), intent(in),optional :: C
  double precision, allocatable :: AB(:,:)

  logical,optional :: ta,tb
  integer :: m,n,k,lda,ldb,nc,mc
  double precision :: alpha
  character(1) :: transa,transb

    m = size(A,1)
    n = size(B,2)
    k = size(A,2)
    transa = 'n'
    transb = 'n'
    lda = size(A,1)
    ldb = size(B,1)
    alpha = 0.
    nc = size(A,1)
    mc = size(B,2)


    if (present(ta)) then
        if(ta) then
            transa = 't'
            lda = size(A,2)
            nc = size(A,2)
            k = size(A,1)
        endif
    endif

    if (present(tb)) then
        if(tb) then
            transb = 't'
            ldb = size(B,2)
            mc = size(B,1)
        endif
    endif

    if (present(C)) then
        AB = C
        alpha = 1.
    else
        if (allocated(AB)) deallocate(AB)
        allocate(AB(nc,mc))
        AB=0.
    endif
    
    call dGEMM ( transa, transb, nc, mc, k, 1.d0, A, lda, &
                       B, ldb, alpha, AB, nc )
                       
end function matmul

! Returns the inverse of a matrix calculated by finding the LU
! decomposition.  Depends on LAPACK.
function inv(A) result(Ainv)
implicit none
  double precision, dimension(:,:), intent(in) :: A
  double precision, allocatable :: Ainv(:,:)

  double precision, allocatable :: work(:)  ! work array for LAPACK
  integer, allocatable :: ipiv(:)   ! pivot indices
  integer :: n, info


  allocate(Ainv(size(A,1),size(A,2)),work(size(A,1)),ipiv(size(A,1)))


  ! Store A in Ainv to prevent it from being overwritten by LAPACK
  Ainv = A
  n = size(A,1)

  ! DGETRF computes an LU factorization of a general M-by-N matrix A
  ! using partial pivoting with row interchanges.
  call dGETRF(n, n, Ainv, n, ipiv, info)

  if (info /= 0) then
     stop 'Matrix is numerically singular!'
  end if

  ! DGETRI computes the inverse of a matrix using the LU factorization
  ! computed by DGETRF.
  call dGETRI(n, Ainv, n, ipiv, work, n, info)

  if (info /= 0) then
     stop 'Matrix inversion failed!'
  end if
  
  deallocate(work,ipiv)
end function inv

end subroutine kalman_apply_f

subroutine gauss_f(mu, dev, out, n)
#ifdef __INTEL_COMPILER
use IFPORT
#endif
implicit none
  integer,intent(in) :: n
  double precision, dimension(n), intent(in) :: mu
  double precision, dimension(n), intent(out) :: out
  double precision, intent(in) :: dev
  double precision,parameter :: TWOPI = 2.*acos(-1.)
  double precision,allocatable :: x2pi(:),g2rad(:)
  logical,save :: init = .false.
  integer,parameter :: myseed = 86456
  integer :: m,i

  if(.not.init) then
     call srand(myseed)
     init = .true.
  endif

    m = ceiling(n/2.)
    allocate(x2pi(m),g2rad(m))
      do i = 1,m
        x2pi(i) = rand() * TWOPI
        g2rad(i) = sqrt(-2.d0 * log(1.d0 - rand()))
      enddo
      do i = 1,size(mu),2
        out(i) = mu(i) + cos(x2pi(ceiling(i/2.))) * g2rad(ceiling(i/2.)) * dev
      enddo
      do i = 2,size(mu),2
        out(i) = mu(i) + sin(x2pi(ceiling(i/2.))) * g2rad(ceiling(i/2.)) * dev
      enddo
      deallocate(x2pi,g2rad)
end subroutine gauss_f
