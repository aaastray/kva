import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('@/views/PatientsList.vue'),
    },
    {
      path: '/add-patient',
      name: 'addPatient',
      component: () => import('@/views/AddPatient.vue')
    },
    {
      path: '/patient/:id',
      name: 'patient',
      component: () => import('@/views/PatientDetail.vue'),
      props: true
    }
  ],
})

export default router
