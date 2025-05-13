<template>
  <div class="patients-list">
    <div class="patients-header">
      <h2>Архив пациентов</h2>
      <el-button type="primary" @click="goToAddPatient">
        <el-icon class="custom-icon"><Plus /></el-icon>
        <span>Добавить пациента</span>
      </el-button>
    </div>

    <el-card class="filter-card">
      <div class="filter-container">
        <el-input
            v-model="filter.searchQuery"
            placeholder="Поиск по ФИО или номеру полиса"
            :prefix-icon="Search"
            clearable
            class="search-input"
        />
      </div>
    </el-card>

    <div class="table-container">
      <el-table
          v-loading="loading"
          :data="filteredPatients"
          style="width: 100%; margin-top: 20px"
          @row-click="handleRowClick"
      >
        <el-table-column prop="lastName" label="Фамилия" min-width="120" />
        <el-table-column prop="firstName" label="Имя" min-width="120" />
        <el-table-column prop="middleName" label="Отчество" min-width="120" />
        <el-table-column prop="birthDate" label="Дата рождения" min-width="150">
          <template #default="{ row }">
            {{ formatDate(row.birthDate) }}
          </template>
        </el-table-column>
        <el-table-column prop="gender" label="Пол" min-width="100">
          <template #default="{ row }">
            {{ row.gender === 'male' ? 'Мужской' : 'Женский' }}
          </template>
        </el-table-column>
        <el-table-column prop="policyNumber" label="Номер полиса" min-width="150" />
        <el-table-column label="Действия" width="120" fixed="right">
          <template #default="{ row }">
            <el-button
                type="primary"
                size="small"
                @click.stop="goToPatientDetail(row.id)"
            >
              Просмотр
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-empty
        v-if="filteredPatients.length === 0"
        description="Нет данных о пациентах"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { usePatientsStore } from '@/stores/patients';
import { type PatientFilter } from '@/types';
import { Search, Plus } from '@element-plus/icons-vue';

const router = useRouter();
const patientsStore = usePatientsStore();
const loading = ref(false);

const filter = ref<PatientFilter>({
  searchQuery: ''
});

const filteredPatients = computed(() => {
  const query = filter.value.searchQuery.toLowerCase().trim();
  if (!query) return patientsStore.patients;

  return patientsStore.patients.filter(patient => {
    const fullName = `${patient.lastName} ${patient.firstName} ${patient.middleName}`.toLowerCase();
    return fullName.includes(query) || patient.policyNumber.toLowerCase().includes(query);
  });
});

const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('ru-RU');
};

const goToPatientDetail = (id: number) => {
  router.push({ name: 'patient', params: { id } });
};

const goToAddPatient = () => {
  router.push({ name: 'addPatient' });
};

const handleRowClick = (row: any) => {
  goToPatientDetail(row.id);
};

onMounted(() => {
  if (patientsStore.patients.length === 0) {
    loading.value = true;
    setTimeout(() => {
      patientsStore.initDemoData();
      loading.value = false;
    }, 500);
  }
});
</script>

<style scoped>
.patients-list {
  padding: 10px 0;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
}

.patients-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-container {
  display: flex;
  gap: 15px;
}

.table-container {
  width: 100%;
  overflow-x: auto;
}
</style>