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
            @input="debouncedFetchPatients"
            @clear="fetchPatients"
        />
      </div>
    </el-card>

    <div class="table-container">
      <el-table
          v-loading="loading"
          :data="patients"
          style="width: 100%; margin-top: 20px"
          @row-click="handleRowClick"
          empty-text="Нет данных о пациентах"
      >
        <el-table-column prop="lastName" label="Фамилия" min-width="120" />
        <el-table-column prop="firstName" label="Имя" min-width="120" />
        <el-table-column prop="middleName" label="Отчество" min-width="120">
            <template #default="{ row }">
                {{ row.middleName || '-' }}
            </template>
        </el-table-column>
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
        v-if="!loading && patients.length === 0 && !filter.searchQuery"
        description="Нет данных о пациентах. Добавьте первого."
    />
     <el-empty
        v-if="!loading && patients.length === 0 && filter.searchQuery"
        description="Пациенты по вашему запросу не найдены."
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import { Search, Plus } from '@element-plus/icons-vue';
import apiClient from '@/services/api'; // Убедитесь, что путь правильный

// Типы данных, которые приходят с бэкенда
interface AnalysisBackend {
  id: number;
  date: string;
  predictedAge: number;
  xrayImageURL: string;
  doctorNotes?: string;
}

interface PatientBackend {
  id: number;
  lastName: string;
  firstName: string;
  middleName?: string;
  birthDate: string; // YYYY-MM-DD
  gender: 'male' | 'female';
  policyNumber: string;
  analyses: AnalysisBackend[];
}

const router = useRouter();
const loading = ref(true);
const patients = ref<PatientBackend[]>([]);

const filter = ref({
  searchQuery: ''
});

let debounceTimer: number | undefined;

const fetchPatients = async () => {
  loading.value = true;
  try {
    const params: { searchQuery?: string } = {};
    if (filter.value.searchQuery.trim()) {
      params.searchQuery = filter.value.searchQuery.trim();
    }
    // Указываем тип ответа для apiClient.get
    const response = await apiClient.get<PatientBackend[]>('/patients', { params });
    patients.value = response.data;
  } catch (error: any) {
    ElMessage.error(`Не удалось загрузить список пациентов: ${error.message || 'Ошибка сервера'}`);
    patients.value = [];
  } finally {
    loading.value = false;
  }
};

const debouncedFetchPatients = () => {
  clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(() => {
    fetchPatients();
  }, 300);
};

const formatDate = (dateString: string): string => {
  if (!dateString) return '-';
  try {
    const [year, month, day] = dateString.split('-');
    if (year && month && day) {
        return `${day}.${month}.${year}`;
    }
    // Если формат другой, пытаемся распарсить как дату
    const date = new Date(dateString);
    if (!isNaN(date.getTime())) {
        return date.toLocaleDateString('ru-RU');
    }
    return dateString; // Возвращаем как есть, если не удалось отформатировать
  } catch (e) {
      console.warn("Could not format date:", dateString, e);
      return dateString;
  }
};

const goToPatientDetail = (id: number) => {
  router.push({ name: 'patient', params: { id } });
};

const goToAddPatient = () => {
  router.push({ name: 'addPatient' });
};

const handleRowClick = (row: PatientBackend) => {
  goToPatientDetail(row.id);
};

onMounted(() => {
  fetchPatients();
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