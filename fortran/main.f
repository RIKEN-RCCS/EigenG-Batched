      program main
      integer, parameter :: n = 33
      real(8) :: a(n,n), z(n,n), d(n), e(n), e2(n)
      integer :: i

      do i=1,n
        do j=1,n
          a(i,j) = min(j,i)
          z(i,j) = 0
        enddo
        z(i,i) = 1
      enddo

      call tred1(n,n,a,d,e,e2)
      call imtql2(n,n,d,e,z,i)
      e2(1:n) = e(1:n)
      call trbak1(n,n,a,e,n,z)

      end

