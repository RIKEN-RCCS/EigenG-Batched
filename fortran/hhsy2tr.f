      subroutine hhsy2tr(nm,n,a,d,e,mb)

      implicit NONE
      integer, intent(in)    :: nm, n, mb
      real(8), intent(inout) :: a(1:nm,1:n),d(1:n),e(1:n)

      integer :: i,j,k,l,m
      integer :: ib,i0,i1,i2,m0,m1,ib0,ib1
      real(8) :: f,g,h,s,ajk
      real(8) :: alpha,beta,anorm2

      real(8) :: u(1:n,1:mb), v(1:n,1:mb), scale(1:mb)

      real(8), parameter :: ZERO = 0.0d0
      real(8), parameter :: ONE  = 1.0d0


      ib0 = (n-1)/mb+1
      ib1 = max(1,2-mb)

      do ib = ib0, ib1, -1

         i0 = (ib-1)*mb
         i1 = min(i0+mb,n)
         m0 = i1 - i0
         m1 = max(1,2*(2-ib))

         do m = m0, m1, -1
            i = i0 + m
            l = i - 1

            scale(m) = ZERO
            do j = 1, i
               u(j,m) = a(j,i)
            enddo
            do k = m0, m+1, -1
               do j = 1, i
                  u(j,m) = u(j,m) + u(j,k)*v(i,k) + v(j,k)*u(i,k)
               enddo
            enddo

            d(i) = u(i,m)
            e(i) = ZERO
            u(i:i1,m) = ZERO

            s = ZERO
            do j = 1, L
               s = s + ABS(u(j,m))
            enddo
            scale(m) = s

            if (s == ZERO) then
               v(1:i1,m) = ZERO
               cycle
            endif

            anorm2 = ZERO
            do j = 1, L
               u(j,m) = u(j,m) / s
               anorm2 = anorm2 + u(j,m)**2
            enddo
            anorm2 = sqrt(anorm2)

            f = u(L,m)
            g = -sign(anorm2,f)
            f = f - g
            u(L,m) = f
            beta = f * g
            e(i) = s * g

            do k = 1, L
               f = u(k,m)
               g = ZERO
               do j = 1, k-1
                  ajk  = a(j,k)
                  g      = g      + ajk * u(j,m)
                  v(j,m) = v(j,m) + ajk * f
               enddo
               v(k,m) = g + a(k,k) * f
            enddo

            do k = m0, m+1, -1
               f = ZERO
               g = ZERO
               do j = 1, L
                  f = f + v(j,k) * u(j,m)
                  g = g + u(j,k) * u(j,m)
               enddo
               do j = 1, L
                  v(j,m) = v(j,m) + f * u(j,k) + g * v(j,k)
               enddo
            enddo

            alpha = ZERO
            do j = 1, L
               v(j,m) = v(j,m) / beta
               alpha = alpha + v(j,m) * u(j,m)
            enddo
            alpha = alpha/(2*beta)
            do j = 1, L
               v(j,m) = v(j,m) + alpha*u(j,m)
            enddo

            v(i:i1,m) = ZERO

         enddo

         i2 = i0+m1-1
         do k = 1, i2
            do m = m0, m1, -1
               f = u(k,m)
               g = v(k,m)
               do j = 1, k
                  a(j,k) = a(j,k) + f * v(j,m) + g * u(j,m)
               enddo
            enddo
         enddo
         do m = m0, m1, -1
            i = i0 + m
            do j = 1, i
               a(j,i) = u(j,m)*scale(m)
            enddo
         enddo

      enddo

      e(1) = ZERO
      d(1) = a(1,1)
      e(2) = a(1,2)
      d(2) = a(2,2)

      return
      end subroutine hhsy2tr
