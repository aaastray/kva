<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="app-wrapper">
    <el-header>
      <div class="header-content">
        <h1>Костно-возрастной анализатор</h1>
      </div>
    </el-header>
    <el-container class="app-container">
      <el-main>
        <RouterView />
      </el-main>
      <el-footer height="40px">
        <div class="footer-content">
          <p>© {{ new Date().getFullYear() }} Костно-возрастной анализатор</p>
        </div>
      </el-footer>
    </el-container>
  </div>
</template>

<style scoped>
.app-wrapper {
  width: 100%;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-container {
  flex: 1;
  width: 100%;
}

.el-header {
  background-color: #409EFF;
  color: white;
  display: flex;
  align-items: center;
  width: 100vw;
  margin-left: calc(-50vw + 50%);
  margin-right: calc(-50vw + 50%);
  padding-left: calc(50vw - 50%);
  padding-right: calc(50vw - 50%);
}

.header-content {
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
}

.el-footer {
  background-color: #f5f7fa;
  color: #606266;
  text-align: center;
  width: 100vw;
  margin-left: calc(-50vw + 50%);
  margin-right: calc(-50vw + 50%);
  padding-left: calc(50vw - 50%);
  padding-right: calc(50vw - 50%);
}

.footer-content {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
}
</style>